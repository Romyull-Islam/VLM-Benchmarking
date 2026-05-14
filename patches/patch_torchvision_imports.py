"""One-shot helper: wrap top-level `import torchvision...` / `from torchvision...` lines in vlmeval/ with try/except ImportError.

Idempotent: if the line is already inside a try block (preceded by `try:`), skip it.
Only touches lines that start at column 0 (true top-level imports).
"""
import re
from pathlib import Path

FILES = [
    "VLMEvalKit/vlmeval/dataset/mmlongbench.py",
    "VLMEvalKit/vlmeval/dataset/tamperbench.py",
    "VLMEvalKit/vlmeval/dataset/mvbench.py",
    "VLMEvalKit/vlmeval/dataset/utils/SArena/CLIP_Score.py",
    "VLMEvalKit/vlmeval/dataset/utils/SArena/LPIPS.py",
    "VLMEvalKit/vlmeval/dataset/utils/tamperbench.py",
    "VLMEvalKit/vlmeval/dataset/EgoExoBench/utils.py",
    "VLMEvalKit/vlmeval/dataset/utils/SArena/inception.py",
    "VLMEvalKit/vlmeval/dataset/utils/SArena/FID.py",
    "VLMEvalKit/vlmeval/dataset/utils/mvbench.py",
    "VLMEvalKit/vlmeval/dataset/utils/SArena/video/LPIPS_video.py",
    "VLMEvalKit/vlmeval/vlm/vintern_chat.py",
    "VLMEvalKit/vlmeval/vlm/aki.py",
    "VLMEvalKit/vlmeval/vlm/internvl/utils.py",
    "VLMEvalKit/vlmeval/vlm/video_llm/pllava.py",
    "VLMEvalKit/vlmeval/vlm/video_llm/videochat2.py",
    "VLMEvalKit/vlmeval/vlm/ursa/ursa_model/image_processing_vlm.py",
    "VLMEvalKit/vlmeval/vlm/mplug_owl3.py",
    "VLMEvalKit/vlmeval/vlm/ursa/ursa_model/clip_encoder.py",
    "VLMEvalKit/vlmeval/vlm/ristretto.py",
    "VLMEvalKit/vlmeval/vlm/minimonkey.py",
    "VLMEvalKit/vlmeval/vlm/qianfan_vl.py",
    "VLMEvalKit/vlmeval/vlm/nvlm.py",
    "VLMEvalKit/vlmeval/dataset/EgoExoBench/egoexobench.py",
    "VLMEvalKit/vlmeval/vlm/xcomposer/xcomposer2_4KHD.py",
    "VLMEvalKit/vlmeval/vlm/qtunevl/qtune_vl_chat.py",
    "VLMEvalKit/vlmeval/vlm/mmalaya.py",
    "VLMEvalKit/vlmeval/vlm/sail_vl.py",
    "VLMEvalKit/vlmeval/vlm/xcomposer/xcomposer2d5.py",
    "VLMEvalKit/vlmeval/vlm/xcomposer/xcomposer2.py",
]

IMPORT_RE = re.compile(r"^(import\s+torchvision\b|from\s+torchvision\b).*$")
MARKER = "# [jetson] torchvision optional"

def patch_file(path: Path) -> int:
    text = path.read_text()
    if MARKER in text:
        return 0  # already patched
    lines = text.splitlines(keepends=False)
    out = []
    patched = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if IMPORT_RE.match(line):
            # Wrap this single import in try/except.
            out.append(f"try:  {MARKER}")
            out.append(f"    {line}")
            out.append("except ImportError:")
            out.append("    pass")
            patched += 1
        else:
            out.append(line)
        i += 1
    if patched:
        # Trailing newline preserved
        new_text = "\n".join(out) + ("\n" if text.endswith("\n") else "")
        path.write_text(new_text)
    return patched

if __name__ == "__main__":
    total = 0
    for rel in FILES:
        p = Path(rel)
        if not p.exists():
            print(f"  MISSING: {rel}")
            continue
        n = patch_file(p)
        total += n
        print(f"  {rel}: patched {n} imports")
    print(f"Total imports wrapped: {total}")
