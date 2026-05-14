# Custom VLM Benchmark

This repository provides a customizable Python module to benchmark Visual Language Models (VLMs) on popular multimodal datasets.  
It builds on and extends the excellent [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) by OpenCompass, adapting it for lightweight usage, energy-aware setups, and additional flexibility.

## 🔗 Acknowledgment

This project is based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), which provides comprehensive support for benchmarking vision-language models.  
We thank the original authors for their open-source contribution, and this repository reuses and modifies parts of that framework for our custom experiments.

---

## 📦 Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for large models)(for Raspberry pi or cpu only device not mandatory)
- [git](https://git-scm.com/) (to clone VLMEvalKit)

---

## 🔧 Installation

1. **Clone this repository and VLMEvalKit:**

    ```bash
    git clone https://github.com/Romyull-Islam/VLM-Benchmarking.git
    ```

2. **Install dependencies:**

    It's recommended to use a virtual environment (conda or venv):

    ```bash
    pip install -r requirements.txt
    ```

    This will install:
    - PyTorch
    - Huggingface Transformers
    - pandas
    - tqdm
    - Pillow  
    and other dependencies used by VLMEvalKit and this script.

3. **(Optional) Set up CUDA:**

    For best performance:

    ```bash
    nvcc -V  # Verify CUDA version
    ```

    Ensure CUDA 11.7+ and appropriate NVIDIA drivers are installed.

4. **Huggingface token:**

    Create a file named `hf_token.txt` and paste your token inside it (used for gated models).

---

## 🚀 Usage

### Single benchmark run

Run the main script:

```bash
python custom_vlm_benchmark.py
```

Edit the `main()` function in `custom_vlm_benchmark.py` to select:
- **Model:**
    - `InternVL2_5-4B-MPO`
    - `Moondream2`
    - `SmolVLM2-256M`, `SmolVLM2-500M`
- **Dataset:**
    - `MMBench_DEV_EN`
    - `SEEDBench_IMG`
    - `MMStar`
    - `MME`

The script will print results to the console and save them into `./outputs/`.

---

### Multi-run reproducibility experiment (Jetson AGX Orin)

For per-model / per-benchmark reproducibility statistics (mean ± std over
N cold-start runs with energy, time, accuracy, and 1 Hz resource logging),
use the multi-run orchestrator:

```bash
# Single (model, benchmark) sweep — 3 cold-start runs with 5-min cooldowns:
./run_multirun_agx_orin.sh --model Moondream2 --dataset MMBench_DEV_EN \
    --percentage 12 --results-dir results/Moondream2_MMBench

# Full sweep across the 3 paper models × 4 benchmarks (~5–6 h unattended):
FROM_SCRATCH=1 ./run_full_sweep.sh

# After all sweeps complete, build cross-model cross-benchmark tables:
./run_multirun_agx_orin.sh --aggregate results
```

What each script does:

| Script | Purpose |
|---|---|
| `run_multirun_agx_orin.sh` | Single-sweep launcher (one model on one benchmark, N cold-start runs) |
| `multirun_agx_orin.py` | The orchestrator: live 1 Hz jtop logging (per-rail power, temps, CPU/RAM/GPU util), MAXN check, resumability, accuracy parsing, summary writing |
| `multirun_single_run.py` | Cold-start subprocess wrapper around `CustomVLMBenchmark.run_benchmark()` |
| `run_full_sweep.sh` | Batch driver: iterates 3 models × 4 benchmarks and aggregates at the end |
| `smoke_test_moondream.py` | Quick end-to-end check that Moondream2 + MMBench works |

**Outputs per sweep** (`results/<sweep>/`):
- `run_N/jtop_log.csv` — 1 Hz timeseries: per-rail power (mW), temps, CPU per-core load+freq, RAM, GPU util/freq
- `run_N/run_meta.json` — per-run metrics: duration, accuracy, energy (per-rail + total), boundary temps
- `summary.csv` — per-run aggregate (one row per run)
- `summary_stats.txt` — mean ± std + paste-ready paragraph for the paper
- `variance_plot.png` — bar chart with error bars

**Outputs after aggregation** (`results/_aggregate/`):
- `aggregate_table.txt` — paste-ready model × benchmark table (accuracy / energy / time / power)
- `accuracy_pct.csv`, `energy_wh.csv`, `time_s.csv`, `avg_power_w.csv` — pivot CSVs (rows=models, cols=benchmarks)
- `aggregate_metrics.csv` — long-format for pandas analysis

See [MULTIRUN_README.md](MULTIRUN_README.md) for full details (flags, energy method, runtime estimates).

---

## 📓 Jetson-specific notes

If you're running this on a Jetson (AGX Orin / Orin NX), there are
JetPack-specific install quirks documented in [SETUP_NOTES.md](SETUP_NOTES.md):

- The JetPack PyTorch wheel (`torch 2.5.0a0+nv24.08`) must be installed
  manually; `pip install -r requirements.txt` would otherwise replace it
  with a generic PyPI build that doesn't see CUDA. The `torch` line in
  `requirements.txt` is therefore commented out by default.
- A small number of VLMEvalKit source files need patches (decord /
  torchvision become optional imports; vlm-init becomes tolerant of
  individual VLMs missing optional deps). The patches are reproducible
  via the scripts in [`patches/`](patches/) — see [patches/README.md](patches/README.md).
- `libvips42` is required by Moondream2's image preprocessing
  (`sudo apt install -y libvips42`).
- Resource + power monitoring uses `jtop` (jetson-stats); install with
  `sudo pip install -U jetson-stats` and enable the service with
  `sudo systemctl enable --now jetson_stats.service`.

## Notes

- You can use either Huggingface Hub models or local checkpoints by editing the `model_path` in the script.
- Some models and datasets require significant GPU memory.
- VLMEvalKit will automatically download datasets and cache them locally.
- For more models and datasets, extend the `self.supported_models` and `self.supported_datasets` inside the CustomVLMBenchmark class of the script.

## Troubleshooting

- If you encounter CUDA or memory errors, ensure your GPU and drivers are compatible and have enough memory.
- If you see missing dependency errors, re-run `pip install -r requirements.txt`.
- On Jetson, if `torch.cuda.is_available()` is `False`, the JetPack wheel was clobbered — see the one-line restore command in [SETUP_NOTES.md](SETUP_NOTES.md).
- For issues with VLMEvalKit, consult their [GitHub repository](https://github.com/open-compass/VLMEvalKit).
