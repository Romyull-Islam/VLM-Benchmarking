"""Idempotent patcher for vlmeval: wrap top-level `import torchvision...` and `import decord` lines in try/except ImportError.

Skips files already patched (detected by MARKER).
"""
import re
import subprocess
from pathlib import Path

MARKER = "# [jetson] optional dep"
TORCHVISION_RE = re.compile(r"^(import\s+torchvision\b|from\s+torchvision\b).*$")
DECORD_RE = re.compile(r"^import\s+decord\b.*$")

def find_files(pattern, root="VLMEvalKit/vlmeval"):
    out = subprocess.check_output(
        ["grep", "-rln", "-E", pattern, root]
    ).decode().splitlines()
    return [Path(f) for f in out if f]

def patch_file(path: Path, regexes):
    text = path.read_text()
    lines = text.splitlines(keepends=False)
    out = []
    patched = 0
    for line in lines:
        # Skip lines already inside a try block (look back; cheap check: previous out line)
        matched = any(rx.match(line) for rx in regexes)
        if matched and (not out or out[-1].strip() != "try:" and MARKER not in line):
            # Wrap. But only if line is at column 0 (top-level).
            if line == line.lstrip():
                out.append(f"try:  {MARKER}")
                out.append(f"    {line}")
                out.append("except ImportError:")
                out.append("    pass")
                patched += 1
                continue
        out.append(line)
    if patched:
        new_text = "\n".join(out) + ("\n" if text.endswith("\n") else "")
        path.write_text(new_text)
    return patched

if __name__ == "__main__":
    total_tv, total_dc = 0, 0
    # torchvision
    tv_files = find_files(r"^(import|from)\s+torchvision\b")
    for f in tv_files:
        # Check if already patched for this dep
        existing = f.read_text()
        # Use re.search to find unwrapped imports
        has_unwrapped = False
        prev_blank = True
        for line in existing.splitlines():
            if TORCHVISION_RE.match(line) and MARKER not in line:
                has_unwrapped = True
                break
        if has_unwrapped:
            n = patch_file(f, [TORCHVISION_RE])
            total_tv += n
            print(f"  TV {f}: +{n}")
    # decord
    dc_files = find_files(r"^import\s+decord\b")
    for f in dc_files:
        existing = f.read_text()
        has_unwrapped = False
        for line in existing.splitlines():
            if DECORD_RE.match(line):
                has_unwrapped = True
                break
        if has_unwrapped:
            n = patch_file(f, [DECORD_RE])
            total_dc += n
            print(f"  DC {f}: +{n}")
    print(f"torchvision wraps: {total_tv}  decord wraps: {total_dc}")
