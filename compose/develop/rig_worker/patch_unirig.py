"""Two edits to the vendored UniRig clone, both needed for skeleton inference.

Run after `git clone` (see Dockerfile.cuda). Idempotent - safe to re-run, and
re-running is what happens on every rebuild.

WHY PATCH A VENDORED REPO AT ALL
UniRig is a repo driven by shell scripts, not a package, so there is no
supported extension point. Both edits below remove a hard dependency the
SKELETON path does not actually use, and neither changes what the model
computes. Kept as a script rather than a fork so the diff stays visible and a
UNIRIG_REF bump surfaces conflicts loudly instead of silently reverting.

EDIT 1 - make the PTv3 / spconv import optional.
    parse_encoder.py imports PointTransformerV3Object at module scope, which
    pulls in spconv. But the skeleton task
    (quick_inference_skeleton_articulationxl_ar_256.yaml) selects
    `mesh_encoder.__target__: michelangelo_encoder`, so PTv3 is never
    constructed. The import alone aborts inference.
    spconv has no cu130 wheel; spconv-cu120 was measured here to import
    cleanly and then die with SIGFPE (exit 136) on its first kernel launch, so
    substituting the older wheel is not an option and a source build is hours.
    Guarded so a config that DOES ask for ptv3obj still fails loudly.

EDIT 2 - flash_attention_2 -> sdpa.
    The AR model config requests `_attn_implementation: flash_attention_2`,
    so transformers refuses to build the model unless flash_attn is installed.
    flash_attn has no wheel for this torch/CUDA/Python combination and builds
    for hours on a 4-thread i3-8100, with real OOM risk against WSL's 11GB.
    FlashAttention-2 computes EXACT attention - it is a memory/IO optimisation,
    not an approximation - and torch's sdpa is likewise exact (and dispatches to
    flash-style kernels itself where it can). Outputs therefore agree to
    float-rounding. What is lost is speed and peak-memory headroom, neither
    of which binds on a single 350M model over a 1024-token sequence.

EDIT 3 - make the skin model import optional.
    parse.py imports BOTH UniRigAR and UniRigSkin at module scope, and
    unirig_skin.py does `from flash_attn.modules.mha import MHA`. So even with
    edit 2 applied, merely importing the model registry needs flash_attn - for a
    class the skeleton task never instantiates. Same shape as edit 1: an eager
    import of an unused sibling.
    This is the honest boundary of this image: SKINNING does need flash_attn and
    is not reachable without building it. The consistency gate needs skeletons
    only, so that build is deferred, not dismissed.
"""
import pathlib
import sys

UNIRIG = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/UniRig")

PTV3_OLD = """from .pointcept.models.PTv3Object import get_encoder as get_encoder_ptv3obj
from .pointcept.models.PTv3Object import PointTransformerV3Object
"""

PTV3_NEW = '''try:
    from .pointcept.models.PTv3Object import get_encoder as get_encoder_ptv3obj
    from .pointcept.models.PTv3Object import PointTransformerV3Object
except ImportError as _ptv3_error:  # patched: spconv is not installed
    # The skeleton task uses michelangelo_encoder, so this encoder is dead
    # weight there. Anything that genuinely asks for it still gets a clear
    # error rather than a NoneType surprise deeper in.
    _PTV3_ERROR = _ptv3_error
    PointTransformerV3Object = None

    def get_encoder_ptv3obj(*args, **kwargs):
        raise ImportError(
            "the ptv3obj mesh encoder needs spconv, which is not installed in "
            "this image (no cu130 wheel exists; the cu120 wheel SIGFPEs). The "
            "skeleton task does not use it - if you selected it deliberately, "
            "build spconv from source."
        ) from _PTV3_ERROR
'''


SKIN_OLD = """from .unirig_ar import UniRigAR
from .unirig_skin import UniRigSkin
"""

SKIN_NEW = '''from .unirig_ar import UniRigAR

try:
    from .unirig_skin import UniRigSkin
except ImportError as _skin_error:  # patched: flash_attn is not installed
    _SKIN_ERROR = _skin_error

    class UniRigSkin:  # type: ignore[no-redef]
        """Placeholder so importing the registry does not need flash_attn.

        Selecting this model still fails, but at construction with a readable
        message rather than at import with a bare ModuleNotFoundError.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "the unirig_skin model needs flash_attn, which is not "
                "installed in this image - no wheel exists for this "
                "torch/CUDA/Python combination and a source build takes hours. "
                "Skeleton inference does not use it."
            ) from _SKIN_ERROR
'''


def patch(path, old, new, label):
    if not path.exists():
        print(f"SKIP {label}: {path} not found")
        return False
    text = path.read_text(encoding="utf-8")
    if new.strip().splitlines()[0] in text and old not in text:
        print(f"OK   {label}: already patched")
        return True
    if old not in text:
        print(f"FAIL {label}: anchor not found in {path} - UniRig changed "
              f"upstream, re-check this patch")
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"OK   {label}: patched {path}")
    return True


def main():
    ok = True
    ok &= patch(UNIRIG / "src/model/parse_encoder.py",
                PTV3_OLD, PTV3_NEW, "ptv3-optional")
    ok &= patch(UNIRIG / "src/model/parse.py",
                SKIN_OLD, SKIN_NEW, "skin-optional")

    # Every config that requests flash attention, not just the one task, so a
    # different --skeleton_task does not resurrect the dependency.
    hits = 0
    for cfg in (UNIRIG / "configs").rglob("*.yaml"):
        text = cfg.read_text(encoding="utf-8")
        if "flash_attention_2" in text:
            cfg.write_text(text.replace("flash_attention_2", "sdpa"),
                           encoding="utf-8")
            print(f"OK   flash->sdpa: {cfg.relative_to(UNIRIG)}")
            hits += 1
    if not hits:
        print("OK   flash->sdpa: nothing to do (already patched)")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
