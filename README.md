# Custom VLM Benchmark

Benchmarking harness for small Vision-Language Models (VLMs) on popular
multimodal datasets, aimed at edge and energy-aware setups (CPU-only boards,
Jetson devices, and CUDA GPUs).

It builds on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) by
OpenCompass, adapting it for lightweight usage and adding:

- a `CustomVLMBenchmark` class that runs one model on one dataset, optionally
  on a deterministic subset of the data (fixed `random_state=42`);
- CPU-only variants of the runner;
- a multi-run orchestrator for the Jetson AGX Orin that repeats each
  (model, benchmark) pair from a cold start and logs energy, time, accuracy,
  and resource utilization at 1 Hz, reporting mean ± std.

---

## Hardware

| Target | Notes |
|---|---|
| CUDA GPU | Recommended for larger models. CUDA 11.7+ with matching NVIDIA drivers. |
| CPU-only (e.g. Raspberry Pi) | Use the `*_cpu*.py` runners; a GPU is not required. |
| Jetson AGX Orin / Orin NX | JetPack-specific setup in [SETUP_NOTES.md](SETUP_NOTES.md). The multi-run orchestrator targets the AGX Orin in MAXN mode. |

Some models and datasets need significant GPU memory.

## Models and datasets

Models registered in `CustomVLMBenchmark.supported_models` include:

- `InternVL2_5-4B-MPO`
- `Moondream2`
- `SmolVLM2-256M`, `SmolVLM2-500M`

The set differs slightly per script (some also list `InternVL2_5-1B(-MPO)` and
`h2ovl-mississippi-*` variants); check `self.supported_models` in the script you
run. The full sweep in `run_full_sweep.sh` defaults to the three models in the
paper's AGX Orin table: `SmolVLM2-256M`, `SmolVLM2-500M`, `Moondream2`.

Datasets (VLMEvalKit names), downloaded and cached automatically by VLMEvalKit:

- `MMBench_DEV_EN`
- `SEEDBench_IMG`
- `MMStar`
- `MME`
- `MMVet` (listed in the scripts' `supported_datasets`)

To add more, extend `self.supported_models` and `self.supported_datasets` in
the `CustomVLMBenchmark` class.

---

## Setup

Prerequisites: Python 3.8+ and [git](https://git-scm.com/).

1. **Clone with the VLMEvalKit submodule.** VLMEvalKit is pinned as a git
   submodule at `VLMEvalKit/`:

    ```bash
    git clone --recursive https://github.com/Romyull-Islam/VLM-Benchmarking.git
    cd VLM-Benchmarking
    ```

    If you already cloned without `--recursive`:

    ```bash
    git submodule update --init
    ```

2. **Install dependencies** (a conda env or venv is recommended):

    ```bash
    pip install -r requirements.txt
    ```

    This installs Transformers, pandas, tqdm, Pillow and the other runtime
    dependencies, and installs `./VLMEvalKit` in editable mode. The `torch`
    line is commented out in `requirements.txt` (see the Jetson notes below);
    on a non-Jetson machine install PyTorch for your platform first.

3. **(Optional) Check CUDA:**

    ```bash
    nvcc -V
    ```

4. **Hugging Face token:** create `hf_token.txt` containing your token (used
   for gated models). It is listed in `.gitignore`.

### Jetson notes

Running on a Jetson (AGX Orin / Orin NX) has JetPack-specific quirks,
documented in [SETUP_NOTES.md](SETUP_NOTES.md):

- The JetPack PyTorch wheel (`torch 2.5.0a0+nv24.08`) must be installed
  manually; `pip install -r requirements.txt` would otherwise replace it with
  a generic PyPI build that doesn't see CUDA. That is why the `torch` line in
  `requirements.txt` is commented out.
- A few VLMEvalKit source files need patches (decord / torchvision become
  optional imports; VLM init tolerates individual VLMs missing optional
  deps). Apply them inside the submodule with the scripts in
  [`patches/`](patches/); see [patches/README.md](patches/README.md).
- `libvips42` is required by Moondream2's image preprocessing
  (`sudo apt install -y libvips42`).
- Resource and power monitoring uses `jtop` (jetson-stats): install with
  `sudo pip install -U jetson-stats` and enable the service with
  `sudo systemctl enable --now jetson_stats.service`.

---

## How to run

### Single benchmark run

| Script | Use |
|---|---|
| `custom_vlm_benchmark.py` | Default runner (uses CUDA when available); subset by `dataset_percentage` (5–100) |
| `custom_vlm_benchmark_cpu.py` | CPU-only runner; subset by `dataset_percentage` |
| `custom_vlm_benchmark_cpu_100.py` | CPU-only runner; subset by a fixed `dataset_sample_size` |
| `custom_vlm_benchmark_100.py` | Another CPU-patched variant; subset by a fixed `dataset_sample_size` |

Pick the model, dataset and subset size by editing `main()` in the script,
then run it:

```bash
python custom_vlm_benchmark.py
```

You can point `model_path` in the script at a Hugging Face Hub model or a
local checkpoint.

### Multi-run reproducibility experiment (Jetson AGX Orin)

For per-model / per-benchmark statistics (mean ± std over N cold-start runs,
with energy, time, accuracy, and 1 Hz resource logging):

```bash
# One (model, benchmark) sweep: 3 cold-start runs with 5-min cooldowns
./run_multirun_agx_orin.sh --model Moondream2 --dataset MMBench_DEV_EN \
    --percentage 12 --results-dir results/Moondream2_MMBench

# Full sweep: 3 paper models x 4 benchmarks (~5–6 h unattended)
FROM_SCRATCH=1 ./run_full_sweep.sh

# After all sweeps complete, build cross-model, cross-benchmark tables
./run_multirun_agx_orin.sh --aggregate results
```

| Script | Purpose |
|---|---|
| `run_multirun_agx_orin.sh` | Single-sweep launcher (one model on one benchmark, N cold-start runs) |
| `multirun_agx_orin.py` | Orchestrator: live 1 Hz jtop logging (per-rail power, temps, CPU/RAM/GPU util), MAXN check, resumability, accuracy parsing, summary writing |
| `multirun_single_run.py` | Cold-start subprocess wrapper around `CustomVLMBenchmark.run_benchmark()` |
| `run_full_sweep.sh` | Batch driver: iterates 3 models × 4 benchmarks and aggregates at the end |
| `smoke_test_moondream.py` | Quick end-to-end check that Moondream2 + MMBench works |

See [MULTIRUN_README.md](MULTIRUN_README.md) for flags, per-dataset
`--percentage` choices, the energy method, runtime estimates, and the
pre-launch sanity check.

---

## Outputs

**Single runs** write to `./outputs/<model>_<dataset>_<timestamp>/`
(predictions and a `scores_<model>_<dataset>` JSON or XLSX file); a summary is
also printed to the console.

**Each multi-run sweep** (`results/<sweep>/`):

- `run_N/jtop_log.csv`: 1 Hz timeseries of per-rail power (mW), temps, per-core CPU load and frequency, RAM, GPU util/freq
- `run_N/run_meta.json`: per-run metrics (duration, accuracy, energy per rail and total, boundary temps)
- `summary.csv`: one row per run
- `summary_stats.txt`: mean ± std plus a paste-ready paragraph for the paper
- `variance_plot.png`: bar chart with error bars

**After aggregation** (`results/_aggregate/`):

- `aggregate_table.txt`: paste-ready model × benchmark table (accuracy / energy / time / power)
- `accuracy_pct.csv`, `energy_wh.csv`, `time_s.csv`, `avg_power_w.csv`: pivot CSVs (rows = models, cols = benchmarks)
- `aggregate_metrics.csv`: long format for pandas analysis

`outputs/` and `results/` are git-ignored.

---

## Troubleshooting

- CUDA or out-of-memory errors: check that your GPU and drivers are compatible and have enough memory.
- Missing dependency errors: re-run `pip install -r requirements.txt` (but not on a Jetson; see below).
- On a Jetson, if `torch.cuda.is_available()` is `False`, the JetPack wheel was replaced; see the restore command in [SETUP_NOTES.md](SETUP_NOTES.md).
- `VLMEvalKit/` is empty: run `git submodule update --init`.
- For VLMEvalKit issues, see its [GitHub repository](https://github.com/open-compass/VLMEvalKit).

## Acknowledgment

This project is based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit),
which provides comprehensive support for benchmarking vision-language models.
We thank the original authors for their open-source contribution; this
repository reuses and modifies parts of that framework for our experiments.
