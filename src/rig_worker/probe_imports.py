"""Which third-party modules does UniRig's tree import but this image lack?

WHY THIS EXISTS
UniRig's requirements.txt is incomplete: it omits torch_scatter, torch_cluster
and spconv, all of which are imported unconditionally by vendored Pointcept
code and all of which abort inference at module-load time. Discovering them by
running inference means one container round-trip per missing module, each
costing a model load. This walks the source instead and reports every gap in
one pass.

Usage (inside rig_worker):  python /app/probe_imports.py /opt/UniRig
"""
import ast
import importlib.util
import pathlib
import sys


def top_level_imports(root):
    """Every distinct top-level module name imported anywhere under `root`.

    Parsed with ast rather than grepped: a regex over import lines also matches
    them inside strings, comments and `try: import x except ImportError` blocks,
    and misses parenthesised multi-imports.
    """
    names = set()
    for path in pathlib.Path(root).rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import - part of UniRig itself.
                if node.level == 0 and node.module:
                    names.add(node.module.split(".")[0])
    return names


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/UniRig"
    missing = []
    for name in sorted(top_level_imports(root)):
        if name in sys.builtin_module_names:
            continue
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError, ModuleNotFoundError):
            missing.append(name)

    # `src` and friends are UniRig's own packages, resolvable only from its root.
    local = {p.name for p in pathlib.Path(root).iterdir()}
    local |= {p.stem for p in pathlib.Path(root).glob("*.py")}
    external = [m for m in missing if m not in local]

    print("MISSING (external):", " ".join(external) or "(none)")
    print("MISSING (local-ish, ignore):", " ".join(m for m in missing
                                                   if m in local) or "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
