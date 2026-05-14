# VLMEvalKit Patches for Jetson AGX Orin

VLMEvalKit's source needs three small patches to run on this Jetson (JetPack 6.1
/ PyTorch `2.5.0a0+nv24.08`). They are idempotent — safe to re-run.

The patches make a handful of optional dependencies non-fatal at import time:
- `decord` (video-only; no aarch64 wheel)
- `torchvision` (~35 modules import it; only needed for some models)
- per-VLM-class imports in `vlmeval/vlm/__init__.py` (any single VLM whose
  optional dep is missing would otherwise break the whole `import vlmeval`)

A fourth patch (the MME yes/no eval) is applied inline to
`VLMEvalKit/vlmeval/dataset/utils/yorn.py` to make MME tolerant of subset
sampling — described in [SETUP_NOTES.md](../SETUP_NOTES.md).

## How to apply (after a fresh `git clone` of VLMEvalKit)

```bash
cd ~/VLM-Benchmarking
# Clone VLMEvalKit if not already present:
[ -d VLMEvalKit/vlmeval ] || git clone https://github.com/open-compass/VLMEvalKit.git VLMEvalKit
# Apply patches:
python3 patches/patch_torchvision_imports.py    # wraps top-level `import torchvision` in try/except
python3 patches/patch_imports.py                # consolidated decord + torchvision pass
python3 patches/patch_vlm_init.py               # tolerant import of every VLM class
```

After all three run, `python3 -c "from vlmeval.config import supported_VLM"`
should succeed with no errors.

## What each script does

| Script | Target files | Operation |
|---|---|---|
| `patch_torchvision_imports.py` | ~30 files in `vlmeval/dataset/*.py`, `vlmeval/vlm/*.py` | Wraps each top-level `import torchvision...` in `try/except ImportError` |
| `patch_imports.py` | Same files + `decord`-using ones | Idempotent re-run of the above plus `import decord` lines |
| `patch_vlm_init.py` | `vlmeval/vlm/__init__.py` | Wraps each `from .X import Y` in tolerant try/except; failed VLMs get a callable stub so `functools.partial(vlm.Y, …)` in `config.py` doesn't crash |

The scripts emit a marker comment (`# [jetson] …`) on patched lines so they
detect prior application and don't double-patch.

## When you'd re-run them

- After `git pull` inside `VLMEvalKit/` (upstream might have added new
  files that import the same optional deps)
- After a fresh `pip install -e ./VLMEvalKit` (no need; pip doesn't touch
  the source tree, but if you re-clone, yes)
