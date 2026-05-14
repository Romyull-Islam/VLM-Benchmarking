"""Make vlmeval/vlm/__init__.py tolerant of individual VLM module ImportErrors.

For each `from .submod import A, B, C` line (possibly multi-line with parens),
wrap it in try/except ImportError that sets A = B = C = None on failure.
This keeps Moondream2 working even if exotic VLMs need missing deps.

Idempotent via MARKER check.
"""
import re
from pathlib import Path

PATH = Path("VLMEvalKit/vlmeval/vlm/__init__.py")
MARKER = "# [jetson] vlm-init tolerant"


def collapse_multiline(text: str) -> str:
    """Join physical lines that are continuations inside a paren-import statement."""
    out = []
    buf = []
    paren = 0
    for line in text.splitlines():
        if paren > 0:
            buf.append(line)
            paren += line.count("(") - line.count(")")
            if paren == 0:
                out.append(" ".join(s.strip() for s in buf))
                buf = []
        else:
            opens = line.count("(")
            closes = line.count(")")
            if opens > closes and line.lstrip().startswith("from"):
                paren = opens - closes
                buf = [line]
            else:
                out.append(line)
    if buf:
        out.append(" ".join(s.strip() for s in buf))
    return "\n".join(out)


STUB_PROLOGUE = '''
# [jetson] vlm-init tolerant-import shim — substitute a stub class for VLMs that
# fail to import due to missing optional deps (torchvision, decord, etc.).
# functools.partial requires a callable, so we provide a class that raises on instantiation.
class _MissingVLM:
    _name = "?"
    def __init__(self, *a, **kw):
        raise ImportError(
            f"VLM '{self.__class__._name}' is unavailable on this machine "
            f"(an optional dependency failed to import). Patched by /tmp/patch_vlm_init.py."
        )

def _make_stub(name):
    return type(f"_Missing_{name}", (_MissingVLM,), {"_name": name})
'''


def patch():
    text = PATH.read_text()
    if MARKER in text:
        print("Already patched.")
        return
    collapsed = collapse_multiline(text)
    lines = collapsed.splitlines()
    out = []
    # Inject the stub prologue after the initial torch lines
    inserted_prologue = False
    n = 0
    pat = re.compile(r"^from\s+\.([\w.]+)\s+import\s+(.+)$")
    for line in lines:
        if not inserted_prologue and line.startswith("torch.manual_seed"):
            out.append(line)
            out.append(STUB_PROLOGUE)
            inserted_prologue = True
            continue
        m = pat.match(line)
        if m:
            submod = m.group(1)
            imports = m.group(2).strip()
            if imports.startswith("(") and imports.endswith(")"):
                imports = imports[1:-1]
            names = [s.strip() for s in imports.split(",") if s.strip()]
            bound = []
            for nm in names:
                if " as " in nm:
                    bound.append(nm.split(" as ")[1].strip())
                else:
                    bound.append(nm)
            out.append(f"try:  {MARKER}")
            out.append(f"    {line}")
            out.append("except Exception as _e:")
            for b in bound:
                out.append(f"    {b} = _make_stub({b!r})")
            n += 1
        else:
            out.append(line)
    new_text = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    PATH.write_text(new_text)
    print(f"Wrapped {n} imports in {PATH}")


if __name__ == "__main__":
    patch()
