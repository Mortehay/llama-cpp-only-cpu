#!/usr/bin/env python3
"""What model is this GGUF, really? Reads the header, KV block and tensor index.

    python3 scripts/gguf-head.py <path-or-url> [<path-or-url> ...]

WHY THIS EXISTS
`.ai/decisions/0012` found four of a 2026-09 batch of GGUFs MISLABELLED in
`general.architecture` (as `wan`, `qwen_image`, `pig`), and CLAUDE.md states
the rule that followed: **a GGUF's architecture label is not evidence - read
the tensor prefixes and block count before wiring a file.** Until this script
there was no way to do that short of loading the model. Now there is, and it
costs a few MB rather than 10 GB: the header, the KV pairs and the whole
tensor index all live at the START of the file, so an HTTP Range request for
the first 4 MB is enough to identify a 10 GB remote quant.

Takes local paths and https:// URLs interchangeably, so a candidate download
can be compared against what is already on disk. Read-only; downloads nothing
beyond the range it asks for.

WORKED EXAMPLE (2026-10-13). The local `Qwen-Image-2512-Q3_K_M.gguf` is a
third-party build whose publisher was never recorded, and the manifest had to
substitute unsloth's. This is how the substitute was shown to be equivalent
rather than merely plausible - identical on every structural field, differing
only in byte size, i.e. a different quantisation run of the same model:

    gguf v3, 1933 tensors, general.architecture 'qwen_image'
    general.file_type 12 (= Q3_K_M) on both
    transformer_blocks=1920 tensors over 60 block indices (0-59)

Compare `general.file_type` too: it is the quant level, and two files can share
an architecture while being different quants.
"""
import struct
import sys
import urllib.request

# GGUF v3 value types
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, \
    UINT64, INT64, FLOAT64 = range(13)
FIXED = {UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h", UINT32: "<I",
         INT32: "<i", FLOAT32: "<f", BOOL: "<?", UINT64: "<Q", INT64: "<q",
         FLOAT64: "<d"}


class Buf:
    def __init__(self, b):
        self.b, self.i = b, 0

    def take(self, n):
        if self.i + n > len(self.b):
            raise EOFError("range too small; re-request more bytes")
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t in FIXED:
            f = FIXED[t]
            return struct.unpack(f, self.take(struct.calcsize(f)))[0]
        if t == STRING:
            return self.string()
        if t == ARRAY:
            et, n = self.u32(), self.u64()
            # Don't materialise huge arrays (tokenizer vocabs); sample them.
            out = []
            for k in range(n):
                v = self.value(et)
                if k < 3:
                    out.append(v)
            return out + ([f"...({n} items)"] if n > 3 else [])
        raise ValueError(f"unknown gguf value type {t}")


def fetch(url, nbytes):
    if not url.startswith(("http://", "https://")):
        import os
        with open(url, "rb") as fh:
            return fh.read(nbytes), f"local 0-{nbytes-1}/{os.path.getsize(url)}"
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}",
                                               "User-Agent": "gguf-head/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(), r.headers.get("Content-Range", "?")


def report_escalating(url, sizes=(4 << 20, 32 << 20, 128 << 20)):
    """Try a small range first, widen if the metadata does not fit.

    4 MB covers a diffusion transformer comfortably. A chat/brain GGUF does
    NOT fit: its KV block carries the tokenizer vocabulary - the 35B brain has
    54 KV pairs including a ~150k-entry string array - and a STRING array
    cannot be skipped without reading each element's length prefix. So widen
    rather than guess, and say which range finally worked.
    """
    print(f"\n=== {url}")
    last = None
    for n in sizes:
        try:
            return report(url, n)
        except EOFError as e:
            last = e
            print(f"  (metadata exceeds {n >> 20} MB; widening)")
    print(f"  ERROR could not read metadata within {sizes[-1] >> 20} MB: {last}")


def report(url, nbytes=4 << 20):
    raw, crange = fetch(url, nbytes)
    print(f"  content-range: {crange}")
    b = Buf(raw)
    magic = b.take(4)
    if magic != b"GGUF":
        print(f"  NOT A GGUF (magic {magic!r})")
        return
    ver = b.u32()
    n_tensors = b.u64()
    n_kv = b.u64()
    print(f"  gguf v{ver}, {n_tensors} tensors, {n_kv} kv pairs")

    kv = {}
    for _ in range(n_kv):
        k = b.string()
        kv[k] = b.value(b.u32())

    for k in ("general.architecture", "general.name", "general.file_type",
              "general.quantization_version"):
        if k in kv:
            print(f"  {k:34} = {kv[k]!r}")
    for k in sorted(kv):
        if "block_count" in k or "n_layer" in k or "expert" in k:
            print(f"  {k:34} = {kv[k]!r}")

    # Tensor index: name, n_dims, dims..., type, offset
    names, prefixes = [], {}
    for _ in range(n_tensors):
        name = b.string()
        nd = b.u32()
        for _ in range(nd):
            b.u64()
        b.u32()
        b.u64()
        names.append(name)
        prefixes[name.split(".")[0]] = prefixes.get(name.split(".")[0], 0) + 1
    print(f"  tensor-name prefixes: "
          f"{', '.join(f'{k}={v}' for k, v in sorted(prefixes.items())[:8])}")
    blocks = {int(n.split(".")[1]) for n in names
              if n.split(".")[0] in ("transformer_blocks", "blk", "blocks")
              and n.split(".")[1].isdigit()}
    if blocks:
        print(f"  distinct block indices: {len(blocks)} (max {max(blocks)})")
    print(f"  sample tensors: {names[:3]}")


if len(sys.argv) < 2:
    sys.exit(__doc__.strip().splitlines()[2].strip())
for u in sys.argv[1:]:
    try:
        report_escalating(u)
    except Exception as e:
        print(f"  ERROR {type(e).__name__}: {e}")
