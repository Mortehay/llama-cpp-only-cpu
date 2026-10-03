"""The brain: a llama-server child of the worker, started and stopped by the
model gateway. Decision: .ai/decisions/0013-gated-brain.md. Contract:
.ai/specs/something2-text/contract.md.

WHY A CHILD OF THE WORKER, NOT A SERVICE. `llm_engine` (llama.cpp router) used
to load a GGUF onto the card whenever anyone called it, outside the gateway -
including from inside a map job with the image pipeline still cached. On a
12 GB card that is the out-of-memory wedge. Here the brain is just another
gateway model: it holds the card while it is the active model, and every
switch away stops it (tasks._gateway_switch), exactly as the persistent Qwen
process is stopped.

TWO BRAINS, ONE RUNTIME (`BRAINS`). The default is Qwen3.6-35B-A3B, a
mixture-of-experts run HYBRID: attention and KV on the card, routed experts in
host RAM (`--n-cpu-moe`), so the host-RAM preflight matters as much as the
VRAM one. The fast one is Qwen3-VL-8B, fully on the card.

THINKING. Qwen3.6 thinks by default. Every request sends
`chat_template_kwargs.enable_thinking=false`; the VL-8B is an Instruct release
with no thinking mode, and the flag is harmless there.

Safe to import from the API process: no torch, no CUDA, at import time.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

MODELS = os.environ.get("MODELS_DIR_IN_CONTAINER", "/models")
BIN = os.environ.get("LLAMA_SERVER_BIN", "/opt/llama/app/llama-server")
LIBS = os.environ.get("LLAMA_SERVER_LIBS", "/opt/llama/app:/opt/llama/lib")
PORT = int(os.environ.get("BRAIN_PORT", "8099"))
CTX = int(os.environ.get("BRAIN_CTX", "8192"))
# Cold load ceiling. Measured 2026-10-01 from a cold file cache: VL-8B 36.6 s,
# 35B 142.9 s (17.7 GB read off the VHDX); 6-13 s once in page cache.
READY_S = float(os.environ.get("BRAIN_READY_S", "240"))
# One completion's ceiling. A hung child must not wedge the solo worker - the
# lesson audio_engine.ACESTEP_TIMEOUT_S records.
REQUEST_S = float(os.environ.get("BRAIN_REQUEST_S", "180"))
# The child exits on its own after this long without a request, so a brain
# nobody uses does not hold the card (or 18 GB of RAM) until the next switch.
IDLE_S = float(os.environ.get("BRAIN_IDLE_S", "600"))

GIB = 1024 ** 3

# id -> what it takes to run it. `min_free_mb` / `min_ram_gib` are checked
# BEFORE starting: refusing cleanly beats OOMing half way (the same rule as
# audio_engine.GPU_BUDGET_MB and qwen_edit's EDIT_HOST_RAM_NEEDED).
BRAINS: dict[str, dict] = {
    "qwen3.6-35b-a3b": {
        "label": "Qwen3.6-35B-A3B (UD-IQ4_XS, hybrid GPU+RAM)",
        "path": os.environ.get(
            "BRAIN_35B_PATH",
            f"{MODELS}/brain-gguf/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"),
        # All layers on the card except the routed experts of the first N of
        # its 40 layers, which stay in RAM. Measured 2026-10-01 (host-side,
        # 8k context, 4 vCPUs), N -> brain VRAM / decode:
        #   40 -> (VRAM not measured) / 10.7 tok/s   24 -> ~8.3 GB / 18.8
        #   20 -> ~9.7 GB / 21.7
        #   16 -> ~10.9 GB / 27.4 (0.3 GB left on the card - the zero-headroom
        #   state behind the 2026-09 faults, so not that one).
        # 24 keeps ~3 GB of the card free and clears 0011's text-lane gates.
        "args": ["-ngl", "99", "--n-cpu-moe",
                 os.environ.get("BRAIN_35B_CPU_MOE", "24")],
        "min_free_mb": int(os.environ.get("BRAIN_35B_MIN_FREE_MB", "9000")),
        "min_ram_gib": float(os.environ.get("BRAIN_35B_MIN_RAM_GIB", "20")),
        "default": True,
    },
    "qwen3-vl-8b": {
        "label": "Qwen3-VL-8B-Instruct (Q4_K_M, GPU)",
        "path": os.environ.get(
            "BRAIN_8B_PATH",
            f"{MODELS}/image-gguf/qwen-image-2.1/Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        "args": ["-ngl", "99"],
        # Measured 2026-10-01: ~5.5 GB host-side at an 8k context.
        "min_free_mb": int(os.environ.get("BRAIN_8B_MIN_FREE_MB", "6500")),
        "min_ram_gib": float(os.environ.get("BRAIN_8B_MIN_RAM_GIB", "2")),
        "default": False,
    },
}

LABEL_PREFIX = "brain:"


class BrainError(RuntimeError):
    """`kind` is what the API maps to a status code:
    unsupported_schema / truncated -> 422; unavailable -> 503; failed -> 502."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def default_brain() -> str:
    return next(b for b, spec in BRAINS.items() if spec.get("default"))


def roster() -> list[dict]:
    """What /api/text/models lists. Static: never loads anything, so it fits
    something2's 10 s discovery budget."""
    out = [{"id": b, "label": spec["label"], "default": bool(spec.get("default")),
            "thinking": False, "available": os.path.exists(spec["path"])}
           for b, spec in BRAINS.items()]
    return sorted(out, key=lambda e: not e["default"])


def label(brain: str | None) -> str:
    """The model-gateway label of a brain."""
    return LABEL_PREFIX + (brain or default_brain())


def is_label(model: str | None) -> bool:
    return bool(model) and model.startswith(LABEL_PREFIX)


# --- The child -----------------------------------------------------------------
# The worker is --pool=solo: one job at a time, so no lock around this.
# `unreported_load_s`: a load done by the gateway switch (ahead of the job)
# is reported by the FIRST completion after it, so timings.load_s is not 0.0
# on exactly the request that paid for it.
_srv: dict = {"proc": None, "brain": None, "timer": None, "log": None,
              "unreported_load_s": 0.0}


def alive(brain: str | None = None) -> bool:
    p = _srv["proc"]
    return (p is not None and p.poll() is None
            and (brain is None or _srv["brain"] == brain))


def stop(reason: str) -> None:
    """Stop the child and free the card and its RAM. Safe when none runs."""
    p, t = _srv["proc"], _srv["timer"]
    _srv.update(proc=None, brain=None, timer=None)
    if t:
        t.cancel()
    if p is None or p.poll() is not None:
        return
    logger.info("brain: stopping (%s)", reason)
    p.terminate()
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def _arm_idle_timer() -> None:
    if _srv["timer"]:
        _srv["timer"].cancel()
    t = threading.Timer(IDLE_S, stop, args=(f"idle {IDLE_S:.0f}s",))
    t.daemon = True
    t.start()
    _srv["timer"] = t


def _log_tail(n: int = 6) -> str:
    try:
        with open(_srv["log"] or "", "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4000))
            lines = f.read().decode("utf-8", "replace").splitlines()
        return " | ".join(lines[-n:])
    except Exception:
        return ""


def _preflight(brain: str) -> None:
    """Refuse to start without the VRAM and host RAM the brain needs."""
    spec = BRAINS[brain]
    if not os.path.exists(spec["path"]):
        raise BrainError("unavailable", f"brain {brain}: {spec['path']} not on disk")
    try:
        import torch
        if torch.cuda.is_available():
            free_mb = torch.cuda.mem_get_info()[0] / 2 ** 20
            if free_mb < spec["min_free_mb"]:
                raise BrainError("unavailable", (
                    f"brain {brain}: {free_mb:.0f} MB VRAM free, needs "
                    f"{spec['min_free_mb']} MB - something else holds the card"))
    except BrainError:
        raise
    except Exception as e:
        logger.warning("brain: VRAM preflight skipped: %s", e)
    try:
        with open("/proc/meminfo") as f:
            avail = next(int(line.split()[1]) * 1024 for line in f
                         if line.startswith("MemAvailable:"))
        if avail < spec["min_ram_gib"] * GIB:
            raise BrainError("unavailable", (
                f"brain {brain}: {avail / GIB:.1f} GiB RAM available, needs "
                f"{spec['min_ram_gib']:.0f} GiB"))
    except BrainError:
        raise
    except Exception as e:
        logger.warning("brain: RAM preflight skipped: %s", e)


def _get(path: str, timeout: float = 2.0) -> int:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}",
                                    timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def ensure(brain: str) -> float:
    """Start the child for `brain` unless it already serves it. Returns the
    load time in seconds (0.0 when already up). Raises BrainError."""
    if alive(brain):
        _arm_idle_timer()
        return 0.0
    stop(f"switching it to {brain}")
    _preflight(brain)
    spec = BRAINS[brain]
    log = f"/tmp/brain-{brain}.log"
    cmd = [BIN, "--host", "127.0.0.1", "--port", str(PORT), "-m", spec["path"],
           "-c", str(CTX), "--jinja", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
           "--threads", str(os.cpu_count() or 4), "--parallel", "1",
           "--no-webui", *spec["args"]]
    started = time.time()
    with open(log, "wb") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                env={**os.environ, "LD_LIBRARY_PATH": LIBS})
    _srv.update(proc=proc, brain=brain, log=log)
    while time.time() - started < READY_S:
        if proc.poll() is not None:
            tail = _log_tail()
            stop("exited during load")
            raise BrainError("failed", f"brain {brain} exited while loading: {tail}")
        if _get("/health") == 200:
            load_s = round(time.time() - started, 1)
            logger.info("brain: %s ready in %.1fs", brain, load_s)
            _srv["unreported_load_s"] = load_s
            _arm_idle_timer()
            return load_s
        time.sleep(0.5)
    stop(f"not ready within {READY_S:.0f}s")
    raise BrainError("failed", f"brain {brain} not ready within {READY_S:.0f}s: "
                               f"{_log_tail()}")


def complete(brain: str, messages: list[dict], *, schema: dict | None = None,
             temperature: float = 0.7, max_tokens: int = 512) -> dict:
    """One chat completion. Returns {text, json?, usage, timings, finish_reason,
    load_s}. Raises BrainError. Starts the child if needed - the CALLER is
    responsible for the card being free (the gateway, or an eviction)."""
    ensure(brain)
    load_s, _srv["unreported_load_s"] = _srv["unreported_load_s"], 0.0
    body = {"messages": messages, "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": {"enable_thinking": False}}
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "out", "strict": True, "schema": schema}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_S) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        # llama-server answers 400 when it cannot turn the schema into a
        # grammar - the request can never succeed as sent.
        if e.code == 400 and schema is not None:
            raise BrainError("unsupported_schema",
                             f"the brain cannot use this schema: {detail}")
        raise BrainError("failed", f"brain HTTP {e.code}: {detail}")
    except Exception as e:
        tail = _log_tail()
        if not alive(brain):
            stop("died mid-request")
        raise BrainError("failed", f"brain request failed: {e} {tail}".strip())
    finally:
        _arm_idle_timer()

    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason")
    out = {"text": text, "finish_reason": finish, "usage": data.get("usage", {}),
           "timings": {"load_s": load_s, "generate_s": round(time.time() - t0, 2),
                       "decode_tok_s": round((data.get("timings") or {})
                                             .get("predicted_per_second", 0), 1)}}
    if schema is not None:
        if finish == "length":
            raise BrainError("truncated", (
                f"the answer hit max_tokens={max_tokens} before the JSON closed; "
                f"raise max_tokens"))
        try:
            out["json"] = json.loads(text)
        except ValueError as e:
            raise BrainError("failed", f"grammar-constrained output did not parse: {e}")
    return out


def complete_in_job(brain: str, messages: list[dict], *, schema: dict | None = None,
                    temperature: float = 0.7, max_tokens: int = 512,
                    route: str = "internal") -> dict:
    """A brain call from INSIDE a running worker job (regions in a map job).

    Such a job is gated under its image model, and the solo worker cannot wait
    on the gateway's queue from inside a task. So the card is cleared here
    instead - pipelines evicted - the brain runs, and it is stopped before the
    job continues: never brain and pipeline at once (0013). The cost is one
    pipeline reload for whatever draws next.

    Returns complete()'s dict, or {"error": ...}. Never raises: every caller
    has a rule fallback. Writes a ledger row like any other brain call.
    """
    import generations
    import tasks

    started = time.time()
    try:
        tasks._evict_pipelines(f"a brain call ({route})")
        tasks.release_vram_cache("brain call in a job")
        out = complete(brain, messages, schema=schema, temperature=temperature,
                       max_tokens=max_tokens)
        generations.record(
            kind="text", status="done", served_from="generated", route=route,
            prompt=messages[-1]["content"], model=brain,
            duration_ms=(time.time() - started) * 1000,
            params={"reply": (out.get("text") or "")[:32768],
                    "timings": out.get("timings"), "usage": out.get("usage")},
            caller={"principal_name": f"{route} (internal)"})
        return out
    except Exception as e:  # noqa: BLE001 - the caller falls back to rules
        logger.warning("brain in job (%s) failed: %s", route, e)
        generations.record(
            kind="text", status="failed", served_from="generated", route=route,
            prompt=messages[-1]["content"], model=brain, error=str(e),
            duration_ms=(time.time() - started) * 1000,
            caller={"principal_name": f"{route} (internal)"})
        return {"error": str(e)}
    finally:
        stop(f"{route} done")
