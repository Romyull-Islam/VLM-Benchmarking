# Jetson AGX Orin Setup Notes

This file documents the Jetson-specific setup quirks for this repo. The default
`pip install -r requirements.txt` does not produce a working environment on
JetPack 6 because of ARM64 wheel and CUDA-stack incompatibilities.

## TL;DR — do not run `pip install -r requirements.txt` again

That command downgrades the JetPack-built PyTorch wheel to a generic PyPI build
that cannot see the Jetson's CUDA driver. To reset the environment after such a
mistake, run:

```bash
pip install --force-reinstall --no-deps \
  "$HOME/Downloads/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
pip install --force-reinstall --no-deps "numpy==1.26.4"
pip uninstall -y torchvision   # PyPI builds are ABI-incompatible with JetPack torch
```

Then verify:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.5.0a0+872d972e41.nv24.08 True
```

## Why the default pip install breaks things

1. **JetPack torch has a pre-release version string** (`2.5.0a0+872d972e41.nv24.8`).
   pip's resolver prefers stable PyPI `torch 2.5.0` when the constraint
   `torch>=2.1.0` is open-ended, so it silently swaps in the generic build.
   The generic build does not work with Jetson CUDA 12.6.

2. **`decord` has no aarch64 wheels** on PyPI. It is only needed for video
   benchmarks (MMBench is image-only), so we make its imports optional.

3. **PyPI `torchvision` is ABI-incompatible** with JetPack torch
   (`operator torchvision::nms does not exist`). torchvision isn't needed for
   Moondream2/MMBench, so we make its imports optional and leave it uninstalled.

4. **VLMEvalKit eagerly imports ~100 VLM classes** in
   `vlmeval/vlm/__init__.py`. Any single one that needs torchvision/decord/etc.
   would break the entire package import. We patched it to substitute a callable
   stub for VLMs that fail to import, so the registry still works for VLMs whose
   deps are present (Moondream2 included).

## Patches applied to VLMEvalKit

The following files in `VLMEvalKit/` have been modified for Jetson compatibility.
They are recorded here so future `git pull`s in the VLMEvalKit submodule do not
silently lose them:

| File | Patch |
|---|---|
| `VLMEvalKit/requirements.txt` | commented out `decord`, `torch`, `torchvision`, `accelerate`; pinned `numpy<2` |
| `VLMEvalKit/vlmeval/vlm/__init__.py` | wrapped each `from .X import Y` in tolerant try/except with callable-stub fallback |
| `VLMEvalKit/vlmeval/vlm/cambrian_s.py` | `from decord import VideoReader, cpu` made optional |
| `VLMEvalKit/vlmeval/dataset/dsrbench.py` | `import decord` made optional |
| `VLMEvalKit/vlmeval/dataset/sitebench.py` | `import decord` made optional |
| `VLMEvalKit/vlmeval/dataset/stibench.py` | `import decord` made optional |
| `VLMEvalKit/vlmeval/dataset/vsibench.py` | `import decord` made optional |
| `VLMEvalKit/vlmeval/vlm/vlm3r.py` | `import decord` made optional |
| ~30 other `vlmeval/dataset/*.py` and `vlmeval/vlm/*.py` files | top-level `import torchvision*` wrapped in try/except |

The patches were applied by `/tmp/patch_imports.py` and `/tmp/patch_vlm_init.py`
(both idempotent — safe to re-run after a VLMEvalKit update if needed).

## Confirmed working setup

```
torch:        2.5.0a0+872d972e41.nv24.08  (JetPack 6 / CUDA 12.6)
torchvision:  not installed (intentionally — Moondream2 doesn't need it)
numpy:        1.26.4
transformers: 4.52.0
vlmeval:      0.1.0 (editable from ./VLMEvalKit, HEAD af1cbcd)
CUDA:         available, device "Orin"
```

## Sanity check

```bash
python3 -c "
import torch
from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('Moondream2 in registry:', 'Moondream2' in supported_VLM)
print('total models:', len(supported_VLM))
"
```

Expected:
```
torch: 2.5.0a0+872d972e41.nv24.08 cuda: True
Moondream2 in registry: True
total models: 553
```
