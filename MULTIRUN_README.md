# Multi-Run Reproducibility Experiment (AGX Orin, headless)

A 3-run measurement of energy, time, accuracy, and resource utilization for
any model+dataset registered in `CustomVLMBenchmark`, addressing the paper
reviewer's request for mean ± std statistics.

**All measurement is headless via jtop** — works fine over SSH; no GUI
required. jtop reads the same kernel registers the Jetson Power GUI exposes,
so per-rail power numbers are directly comparable to the paper's
GUI-based measurements on Pi 5 / Orin NX.

## How to run

```bash
# Default: 3× Moondream2 on MMBench, 12% (~519 samples), 5-min cooldowns.
./run_multirun_agx_orin.sh

# All four datasets in sequence (~90 min total compute + cooldowns):
./run_multirun_agx_orin.sh --dataset MMBench_DEV_EN --percentage 12 --results-dir results/MMBench
./run_multirun_agx_orin.sh --dataset SEEDBench_IMG --percentage 5  --results-dir results/SEED
./run_multirun_agx_orin.sh --dataset MMStar        --percentage 10 --results-dir results/MMStar
./run_multirun_agx_orin.sh --dataset MME           --percentage 5  --results-dir results/MME
```

Each invocation:
1. verifies `nvpmodel -q` shows MAXN (refuses to start otherwise)
2. starts a jtop background thread that logs everything at 1 Hz to `jtop_log.csv`
3. runs the model 3 times in fresh Python subprocesses (cold model load each time)
4. between runs, sleeps for `--cooldown` seconds (default 300)
5. after each run, integrates `jtop_log.csv` trapezoidally to compute energy /
   avg power per rail, extracts start/end temperatures, parses accuracy from
   VLMEvalKit's scores JSON
6. after all runs complete: writes `summary.csv`, `summary_stats.txt`, `variance_plot.png`

## Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Moondream2` | Any model registered in `CustomVLMBenchmark.supported_models` |
| `--dataset` | `MMBench_DEV_EN` | Any VLMEvalKit dataset name |
| `--percentage` | `12` | Sample percentage (5–100); fixed `random_state=42` |
| `--runs` | `3` | Number of independent repetitions |
| `--cooldown` | `300` | Seconds of idle between runs (0 to disable) |
| `--only-run N` | — | Run just index N (resumability) |
| `--force-no-maxn` | — | Proceed even if MAXN not detected |
| `--results-dir` | `results/multirun_agx_orin` | Output directory |

## Picking `--percentage` per dataset

To match the sample counts you used on the other devices:

| Dataset | `--percentage` | Approx samples |
|---|---|---|
| `MMBench_DEV_EN` | `12` | 519 |
| `SEEDBench_IMG` | `5` | 711 |
| `MMStar` | `10` | 510 |
| `MME` | `5` | 522 |

Subsetting is deterministic (random_state=42), so identical sample sets across runs.

## Outputs

```
results/<results-dir>/
├── run_1/
│   ├── jtop_log.csv          # 1 Hz: power per rail + temps + CPU/RAM/GPU util
│   ├── run_meta.json         # all per-run metrics (start/end/duration/accuracy/energy/temps)
│   ├── outputs/              # CustomVLMBenchmark artifacts (predictions xlsx + scores json)
│   └── done.marker           # presence = this run completed (resumability)
├── run_2/, run_3/ …
├── summary.csv               # per-run metrics in the schema requested by the task
├── summary_stats.txt         # mean ± std + paste-ready paragraph
└── variance_plot.png         # bar chart with error bars (compute energy + time)
```

### `jtop_log.csv` columns (1 row per second)

| Column | Source |
|---|---|
| `timestamp`, `iso_time` | wall-clock |
| `VDD_GPU_SOC_mw` | `jtop.power['rail']['VDD_GPU_SOC']['power']` (mW) |
| `VDD_CPU_CV_mw` | `jtop.power['rail']['VDD_CPU_CV']['power']` |
| `VIN_SYS_5V0_mw` | `jtop.power['rail']['VIN_SYS_5V0']['power']` |
| `VDDQ_VDD2_1V8AO_mw` | `jtop.power['rail']['VDDQ_VDD2_1V8AO']['power']` |
| `tot_power_mw` | `jtop.power['tot']['power']` (jtop's sum) |
| `gpu_temp_c`, `cpu_temp_c`, `tj_temp_c`, `soc0/1/2_temp_c` | `jtop.temperature[…]['temp']` |
| `cpu_load_avg_pct` | average over online cores |
| `cpu_loads_per_core_pct` | `;`-joined (12 entries on Orin) |
| `cpu_freqs_per_core_mhz` | `;`-joined |
| `mem_used_mb`, `mem_total_mb`, `mem_used_pct` | `jtop.memory['RAM']` |
| `gpu_load_pct` | `jtop.gpu['gpu']['status']['load']` (0–100) |
| `gpu_freq_mhz` | `jtop.gpu['gpu']['freq']['cur']` / 1000 |

### `summary.csv` columns

| Column | Source |
|---|---|
| `run_id` | 1, 2, 3 |
| `energy_wh` | trapezoidal integral of (`VDD_GPU_SOC` + `VDD_CPU_CV`) from `jtop_log.csv` |
| `time_s` | wall-clock from inference start to eval done |
| `accuracy_pct` | "Overall" field from VLMEvalKit's scores JSON × 100 |
| `avg_power_w` | mean compute power |
| `start_time`, `end_time` | ISO timestamps |
| `gpu_temp_start_c`, `gpu_temp_end_c` | `gpu_temp_c` at run window boundaries; falls back to `tj_temp_c` when `gpu` sensor is offline |

`run_meta.json` is a superset — also contains `energy_memory_wh`,
`energy_system_wh`, `energy_total_wh`, and per-rail `avg_power_*_w`.

## Energy reporting convention

`summary_stats.txt` reports energy four ways for full transparency:

| Label | Rails summed | Use |
|---|---|---|
| **compute** | `VDD_GPU_SOC` + `VDD_CPU_CV` | Standard paper convention; headline `energy_wh` column |
| **memory** | `VDDQ_VDD2_1V8AO` | LPDDR rail |
| **system** | `VIN_SYS_5V0` | Fans, peripherals, USB |
| **total** | sum of all four | Whole-module draw (equivalent to `jtop.power['tot']['power']`) |

## Approximate runtime

At ~2 it/s for Moondream2 on AGX Orin in MAXN:

| Scope | # samples | Per-run | 3 runs + 2×5-min cooldowns |
|---|---|---|---|
| `MMBench_DEV_EN --percentage 12` (default) | 519 | ~4 min | **~22 min** |
| `SEEDBench_IMG --percentage 5` | 711 | ~6 min | ~28 min |
| `MME --percentage 5` | 522 | ~4 min | ~22 min |
| `MMStar --percentage 10` | 510 | ~4 min | ~22 min |

Whole 4-dataset sweep: ~95 min wall-clock.

## Resumability

A run is complete only when its `done.marker` exists. Restarting the
orchestrator skips runs whose marker is present. To force re-run:

```bash
rm results/<results-dir>/run_2/done.marker
./run_multirun_agx_orin.sh --only-run 2
```

## Sanity check before launching

```bash
nvpmodel -q | head -2                                                # should show MAXN
python3 -c "import torch; print(torch.cuda.is_available())"          # True
python3 -c "from jtop import jtop; print('jtop OK')"                 # jtop OK
systemctl is-active jetson_stats.service                             # active
python3 -c "from vlmeval.config import supported_VLM; print('Moondream2' in supported_VLM)"  # True
```

If `torch.cuda.is_available()` is `False`, see [SETUP_NOTES.md](SETUP_NOTES.md)
for the JetPack PyTorch-wheel restore command.
