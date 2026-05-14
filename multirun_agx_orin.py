"""Multi-run reproducibility experiment on Jetson AGX Orin.

Runs a model (default Moondream2) on a dataset (default MMBench_DEV_EN at 12 %
≈ 519 samples) N times (default 3) with idle cooldowns between runs, while a
single background jtop logger captures everything at 1 Hz:
  - Power per rail (VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0, VDDQ_VDD2_1V8AO)
    plus jtop's tot.power — same rails the Jetson Power GUI exposes.
  - Temperatures (gpu, cpu, tj, soc0/1/2)
  - CPU per-core load + frequency
  - RAM used + total
  - GPU load + frequency

Everything ends up in `run_N/jtop_log.csv`. After each run, the orchestrator
integrates the CSV trapezoidally to produce per-rail energy / mean power /
boundary temps, and writes them into `run_N/run_meta.json`. After all runs
complete, it writes summary.csv, summary_stats.txt, and variance_plot.png.

This is the headless replacement for the Jetson Power GUI workflow — works
fine over SSH; no X display needed.
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

DEFAULT_MODEL = "Moondream2"
DEFAULT_DATASET = "MMBench_DEV_EN"
# ~12 % gives ~519 samples on MMBench_DEV_EN, matching the count used on
# other devices (Pi 5, Orin NX). Per-dataset suggestions via --percentage:
#   MMBench_DEV_EN  12% -> 519     SEEDBench_IMG  5% -> 711
#   MMStar          10% -> 510     MME            5% -> 522
DEFAULT_PERCENTAGE = 12
DEFAULT_RUNS = 3
DEFAULT_COOLDOWN_S = 300

# Power rails on AGX Orin. Same names the Jetson Power GUI shows.
RAIL_GPU_SOC = "VDD_GPU_SOC"
RAIL_CPU_CV = "VDD_CPU_CV"
RAIL_SYS = "VIN_SYS_5V0"
RAIL_MEM = "VDDQ_VDD2_1V8AO"
RAIL_NAMES = (RAIL_GPU_SOC, RAIL_CPU_CV, RAIL_SYS, RAIL_MEM)

HERE = Path(__file__).resolve().parent


# ---------- unified jtop logger ----------

class JtopLogger:
    """Background-thread jtop logger at 1 Hz. Captures power + temperature +
    CPU/RAM/GPU utilization in a single CSV. Headless — works over SSH."""

    CSV_HEADER = [
        "timestamp", "iso_time",
        # Power rails (mW)
        "VDD_GPU_SOC_mw", "VDD_CPU_CV_mw", "VIN_SYS_5V0_mw", "VDDQ_VDD2_1V8AO_mw",
        "tot_power_mw",
        # Temperatures (°C)
        "gpu_temp_c", "cpu_temp_c", "tj_temp_c",
        "soc0_temp_c", "soc1_temp_c", "soc2_temp_c",
        # CPU & memory
        "cpu_load_avg_pct",
        "cpu_loads_per_core_pct",        # ';'-joined per online core
        "cpu_freqs_per_core_mhz",        # ';'-joined
        "mem_used_mb", "mem_total_mb", "mem_used_pct",
        # GPU
        "gpu_load_pct", "gpu_freq_mhz",
    ]

    def __init__(self, log_path, interval_s: float = 1.0):
        self.log_path = Path(log_path)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self._error = None

    def _poll_loop(self):
        from jtop import jtop
        try:
            with open(self.log_path, "w", newline="", buffering=1) as f:
                w = csv.writer(f)
                w.writerow(self.CSV_HEADER)
                with jtop(interval=self.interval_s) as jet:
                    while jet.ok() and not self._stop.is_set():
                        # ---- power ----
                        rails = jet.power.get("rail", {}) or {}
                        tot = jet.power.get("tot", {}) or {}
                        # ---- temperatures ----
                        temps = jet.temperature or {}
                        def temp(name):
                            v = (temps.get(name) or {}).get("temp")
                            return v if (v is not None and v > -200) else ""
                        # ---- CPU per-core ----
                        loads_pct, freqs_mhz = [], []
                        for core in (jet.cpu or {}).get("cpu", []):
                            if not core.get("online"):
                                continue
                            idle = core.get("idle")
                            loads_pct.append(round(100.0 - idle, 1) if idle is not None else "")
                            f_khz = (core.get("freq") or {}).get("cur")
                            freqs_mhz.append(round(f_khz / 1000.0, 1) if f_khz else "")
                        numeric_loads = [v for v in loads_pct if isinstance(v, (int, float))]
                        cpu_avg = round(sum(numeric_loads) / len(numeric_loads), 1) if numeric_loads else ""
                        # ---- memory ----
                        try:
                            ram = jet.memory["RAM"]
                            mem_used_mb = round(ram["used"] / 1024.0, 1)
                            mem_tot_mb = round(ram["tot"] / 1024.0, 1)
                            mem_pct = round(100.0 * ram["used"] / ram["tot"], 1) if ram["tot"] else ""
                        except (KeyError, TypeError):
                            mem_used_mb = mem_tot_mb = mem_pct = ""
                        # ---- GPU ----
                        try:
                            gpu_load = jet.gpu["gpu"]["status"]["load"]
                            gpu_freq_khz = jet.gpu["gpu"]["freq"]["cur"]
                            gpu_freq_mhz = round(gpu_freq_khz / 1000.0, 1) if gpu_freq_khz else ""
                        except (KeyError, TypeError):
                            gpu_load = ""
                            gpu_freq_mhz = ""

                        w.writerow([
                            time.time(),
                            datetime.now().isoformat(timespec="seconds"),
                            (rails.get(RAIL_GPU_SOC, {}) or {}).get("power", ""),
                            (rails.get(RAIL_CPU_CV, {}) or {}).get("power", ""),
                            (rails.get(RAIL_SYS, {}) or {}).get("power", ""),
                            (rails.get(RAIL_MEM, {}) or {}).get("power", ""),
                            tot.get("power", ""),
                            temp("gpu"), temp("cpu"), temp("tj"),
                            temp("soc0"), temp("soc1"), temp("soc2"),
                            cpu_avg,
                            ";".join(str(v) for v in loads_pct),
                            ";".join(str(v) for v in freqs_mhz),
                            mem_used_mb, mem_tot_mb, mem_pct,
                            gpu_load, gpu_freq_mhz,
                        ])
                        if self._stop.wait(self.interval_s):
                            break
        except Exception as e:
            self._error = e

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        time.sleep(1.2)  # ensure first sample lands before benchmark starts

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._error:
            print(f"  WARN: jtop logger raised: {self._error!r}")

    # ----- integration -----

    @staticmethod
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def integrate(self):
        """Read the CSV; compute per-rail energy using the SAME method as the
        paper's analysis script:

            total_power_W = sum of all -IN_POWER columns (per sample)
            energy_Wh     = total_power_W / 3600  (per sample, assuming 1 s spacing)
            total_energy  = sum(energy_Wh) over all samples in the window

        Implementation = simple rectangular integration treating each sample
        as a 1-second slice. This matches the original code exactly and keeps
        AGX Orin numbers directly comparable to Pi 5 / Orin NX figures already
        in the paper.

        Returns a dict with per-rail and combined energy + boundary temps."""
        rows = []
        with open(self.log_path, newline="") as f:
            for row in csv.DictReader(f):
                t = self._f(row.get("timestamp"))
                if t is None:
                    continue
                rows.append((t, row))
        out = {"n_power_samples": len(rows)}
        if len(rows) < 1:
            for k in ("compute", "memory", "system", "total"):
                out[f"energy_{k}_wh"] = 0.0
                out[f"avg_power_{k}_w"] = 0.0
            out["duration_s"] = 0.0
            out["gpu_temp_start_c"] = None
            out["gpu_temp_end_c"] = None
            return out

        out["duration_s"] = (rows[-1][0] - rows[0][0]) if len(rows) > 1 else 0.0

        def gpu_soc(r): return (self._f(r.get("VDD_GPU_SOC_mw")) or 0.0) / 1000.0
        def cpu_cv(r):  return (self._f(r.get("VDD_CPU_CV_mw")) or 0.0) / 1000.0
        def sys_5v(r):  return (self._f(r.get("VIN_SYS_5V0_mw")) or 0.0) / 1000.0
        def mem_mw(r):  return (self._f(r.get("VDDQ_VDD2_1V8AO_mw")) or 0.0) / 1000.0

        # Per-sample power series in W (matches paper's df['total_power_W'] exactly).
        compute_w = [gpu_soc(r) + cpu_cv(r) for _, r in rows]
        memory_w  = [mem_mw(r) for _, r in rows]
        system_w  = [sys_5v(r) for _, r in rows]
        total_w   = [gpu_soc(r) + cpu_cv(r) + sys_5v(r) + mem_mw(r) for _, r in rows]

        def rect_energy_wh(values_w):
            # Paper's method: sum(per-sample W) * 1s / 3600 = Wh
            return sum(values_w) / 3600.0

        def mean_w(values_w):
            return sum(values_w) / len(values_w) if values_w else 0.0

        out["energy_compute_wh"] = rect_energy_wh(compute_w)
        out["energy_memory_wh"]  = rect_energy_wh(memory_w)
        out["energy_system_wh"]  = rect_energy_wh(system_w)
        out["energy_total_wh"]   = rect_energy_wh(total_w)
        out["avg_power_compute_w"] = mean_w(compute_w)
        out["avg_power_memory_w"]  = mean_w(memory_w)
        out["avg_power_system_w"]  = mean_w(system_w)
        out["avg_power_total_w"]   = mean_w(total_w)

        # Boundary temperatures — gpu sensor falls back to tj if it reports -256
        def boundary_temp(idx):
            r = rows[idx][1]
            t = self._f(r.get("gpu_temp_c"))
            if t is None or t < 0:
                t = self._f(r.get("tj_temp_c"))
            return t

        out["gpu_temp_start_c"] = boundary_temp(0)
        out["gpu_temp_end_c"] = boundary_temp(-1)
        return out


# ---------- power-mode helper ----------

def verify_maxn():
    try:
        r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
        out = (r.stdout or "") + (r.stderr or "")
        return ("MAXN" in out.upper(), out.strip())
    except Exception as e:
        return (False, f"nvpmodel query failed: {e}")


# ---------- single run orchestration ----------

def cooldown(seconds: int):
    print(f"  cooling down {seconds}s ...")
    end = time.time() + seconds
    while time.time() < end:
        remaining = int(end - time.time())
        print(f"  {remaining:>4}s left", flush=True)
        time.sleep(min(15, max(remaining, 1)))


def run_one(i: int, total: int, run_dir: Path, args) -> dict:
    scope = (f"n={args.n_samples}" if args.n_samples is not None
             else f"@{args.percentage}%")
    print(f"\n=== run {i}/{total} ({args.model} on {args.dataset} {scope}) ===")
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    meta_path = run_dir / "run_meta.json"
    jtop_log_path = run_dir / "jtop_log.csv"

    logger = JtopLogger(jtop_log_path, interval_s=1.0)
    logger.start()

    t0 = time.time()
    try:
        cmd = [sys.executable, str(HERE / "multirun_single_run.py"),
               "--model", args.model,
               "--dataset", args.dataset,
               "--work-dir", str(outputs_dir),
               "--meta-out", str(meta_path)]
        if args.n_samples is not None:
            cmd += ["--n-samples", str(args.n_samples)]
        else:
            cmd += ["--percentage", str(args.percentage)]
        r = subprocess.run(cmd, cwd=str(HERE))
        ok = (r.returncode == 0)
    finally:
        t1 = time.time()
        logger.stop()

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"error": f"child exit {r.returncode}; no meta written"}

    # accuracy
    accuracy = None
    scores_file = meta.get("scores_file")
    if scores_file and Path(scores_file).exists():
        accuracy = _parse_accuracy(Path(scores_file))
    if accuracy is None:
        sf = _find_scores_file(outputs_dir)
        if sf:
            accuracy = _parse_accuracy(sf)

    # power / temp from jtop CSV
    e = logger.integrate()

    meta.update({
        "run_id": i,
        "duration_s": t1 - t0,
        "accuracy": accuracy,
        "accuracy_pct": (100.0 * accuracy) if accuracy is not None else None,
        # Headline (paper-facing) metric: TOTAL of all -IN_POWER rails,
        # rectangular integration (matches the analysis code used for the
        # other devices in the paper).
        "energy_wh": e["energy_total_wh"],
        "avg_power_w": e["avg_power_total_w"],
        # Per-rail breakdown for transparency.
        "energy_compute_wh": e["energy_compute_wh"],
        "energy_memory_wh": e["energy_memory_wh"],
        "energy_system_wh": e["energy_system_wh"],
        "energy_total_wh": e["energy_total_wh"],
        "avg_power_compute_w": e["avg_power_compute_w"],
        "avg_power_memory_w": e["avg_power_memory_w"],
        "avg_power_system_w": e["avg_power_system_w"],
        "avg_power_total_w": e["avg_power_total_w"],
        "n_power_samples": e["n_power_samples"],
        "logger_duration_s": e["duration_s"],
        "gpu_temp_start_c": e["gpu_temp_start_c"],
        "gpu_temp_end_c": e["gpu_temp_end_c"],
    })
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    # done.marker policy: write if the inference subprocess completed cleanly,
    # even if accuracy couldn't be parsed (the energy/time data is still
    # valuable, and we don't want to re-run a working benchmark just because
    # of a scores-file shape we don't recognize). User can always force
    # re-run with `rm done.marker` + --only-run.
    if ok:
        (run_dir / "done.marker").touch()

    if ok and accuracy is not None:
        status = "OK"
    elif ok:
        status = "OK_NO_ACC"   # benchmark finished but score parsing failed
    else:
        status = "FAILED"      # inference subprocess crashed
    print(f"  [{status}] time={t1 - t0:.0f}s  energy_total={e['energy_total_wh']:.2f}Wh  "
          f"avg_total_W={e['avg_power_total_w']:.1f}  acc={accuracy}  "
          f"jtop_samples={e['n_power_samples']}")
    return meta


def _find_scores_file(work_dir: Path):
    """CustomVLMBenchmark writes the eval results either as `scores_*.json`
    (when dataset.evaluate returns a dict) or `scores_*.xlsx` (DataFrame).
    Different VLMEvalKit datasets pick different formats:
      - MMBench / MMVet → dict → .json
      - MMStar / MME    → DataFrame → .xlsx
    Return the most recent of either type."""
    candidates = sorted(
        list(work_dir.glob("*/scores_*.json")) + list(work_dir.glob("*/scores_*.xlsx"))
    )
    return candidates[-1] if candidates else None


def _parse_accuracy(scores_path: Path):
    """Best-effort overall-accuracy extraction. Handles both VLMEvalKit output
    shapes (nested JSON dict, or DataFrame saved as xlsx)."""
    suffix = scores_path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(scores_path.read_text())
            return _walk_for_overall(data)
        elif suffix in (".xlsx", ".xls"):
            return _xlsx_overall(scores_path)
    except Exception as e:
        print(f"  WARN: could not parse {scores_path.name}: {e!r}")
    return None


def _walk_for_overall(x):
    if isinstance(x, dict):
        if "Overall" in x and isinstance(x["Overall"], (int, float)):
            return float(x["Overall"])
        for v in x.values():
            r = _walk_for_overall(v)
            if r is not None:
                return r
    if isinstance(x, dict) and len(x) == 1:
        v = next(iter(x.values()))
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _xlsx_overall(scores_path: Path):
    """Try a few common shapes:
      (a) MMStar-style: index column = subcategory, columns include 'Overall'
      (b) DataFrame with 'split' column and an 'Overall' row
      (c) Single-cell value
    Returns the 'Overall' / first-row average / first numeric cell, in [0,1]."""
    import pandas as pd
    df = pd.read_excel(scores_path)
    # Strategy 1: a column literally named 'Overall'
    for col in df.columns:
        if str(col).strip().lower() == "overall":
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals):
                v = float(vals.iloc[0])
                return v / 100.0 if v > 1.0 else v
    # Strategy 2: index includes 'Overall' (after read_excel with index col)
    for col in df.columns:
        col_data = df[col]
        # Look for a row where any cell equals 'Overall'
        for i, row in df.iterrows():
            for k in row.index:
                if str(row[k]).strip().lower() == "overall":
                    # adjacent numeric in same row
                    for k2 in row.index:
                        try:
                            v = float(row[k2])
                            return v / 100.0 if v > 1.0 else v
                        except (TypeError, ValueError):
                            continue
        break
    # Strategy 3: first numeric cell (last resort)
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals):
            v = float(vals.iloc[0])
            return v / 100.0 if v > 1.0 else v
    return None


# ---------- summary / stats / plot ----------

def _write_summary_csv(rows, csv_path: Path):
    cols = ["run_id", "energy_wh", "time_s", "accuracy_pct", "avg_power_w",
            "start_time", "end_time", "gpu_temp_start_c", "gpu_temp_end_c"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "run_id": r.get("run_id"),
                "energy_wh": r.get("energy_wh"),
                "time_s": r.get("duration_s"),
                "accuracy_pct": r.get("accuracy_pct"),
                "avg_power_w": r.get("avg_power_w"),
                "start_time": r.get("start_iso") or r.get("start_time"),
                "end_time": r.get("end_iso") or r.get("end_time"),
                "gpu_temp_start_c": r.get("gpu_temp_start_c"),
                "gpu_temp_end_c": r.get("gpu_temp_end_c"),
            })
    print(f"  wrote {csv_path}")


def _msd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    rsd = (100.0 * sd / m) if m else 0.0
    return m, sd, rsd


def _write_stats(rows, results_dir: Path, args):
    em, es, er = _msd([r.get("energy_compute_wh") for r in rows])
    mm, ms_, mr = _msd([r.get("energy_memory_wh") for r in rows])
    sm, ss_, sr = _msd([r.get("energy_system_wh") for r in rows])
    tem, tes, ter = _msd([r.get("energy_total_wh") for r in rows])
    tm, ts_, tr = _msd([r.get("duration_s") for r in rows])
    am, as_, ar = _msd([r.get("accuracy_pct") for r in rows])
    pm, ps_, pr = _msd([r.get("avg_power_compute_w") for r in rows])

    # Headline figure for the paper = total of all rails (matches the analysis
    # script used for Pi 5 / Orin NX). pm/ps_/pr now refer to TOTAL avg power.
    pm, ps_, pr = _msd([r.get("avg_power_total_w") for r in rows])

    scope_str = (f"n_samples={args.n_samples}" if args.n_samples is not None
                 else f"{args.percentage}% of samples")

    def f(v, spec=".3f"):
        """Format helper: prints 'N/A' instead of crashing on None.
        Used so old runs without energy data still produce a summary file."""
        return f"{v:{spec}}" if v is not None else "N/A"

    lines = [
        f"Multi-run reproducibility — {args.model} on {args.dataset} "
        f"({scope_str})  n={len(rows)}",
        "=" * 72,
        f"  time_s                                          :  mean={f(tm,'.2f')}   std={f(ts_,'.2f')}   RSD={f(tr,'.2f')}%",
        f"  accuracy_pct                                    :  mean={f(am,'.3f')}  std={f(as_,'.3f')}  RSD={f(ar,'.2f')}%",
        f"  energy_wh, TOTAL  (sum of all -IN_POWER rails)  :  mean={f(tem,'.3f')}  std={f(tes,'.3f')}  RSD={f(ter,'.2f')}%   <-- HEADLINE (matches paper method)",
        f"  energy_wh, compute  (VDD_GPU_SOC + VDD_CPU_CV)  :  mean={f(em,'.3f')}  std={f(es,'.3f')}  RSD={f(er,'.2f')}%",
        f"  energy_wh, memory   (VDDQ_VDD2_1V8AO)           :  mean={f(mm,'.3f')}  std={f(ms_,'.3f')}  RSD={f(mr,'.2f')}%",
        f"  energy_wh, system   (VIN_SYS_5V0)               :  mean={f(sm,'.3f')}  std={f(ss_,'.3f')}  RSD={f(sr,'.2f')}%",
        f"  avg_power_w (TOTAL)                             :  mean={f(pm,'.3f')}  std={f(ps_,'.3f')}  RSD={f(pr,'.2f')}%",
        "",
        "Paper-ready paragraph:",
    ]
    if tem is None:
        para = (
            f"Re-ran {args.model} on {args.dataset} ({scope_str}) {len(rows)} "
            f"times on the Jetson AGX Orin (MAXN). Wall-clock time was "
            f"{f(tm,'.1f')} ± {f(ts_,'.1f')} s (RSD {f(tr,'.2f')}%) and benchmark "
            f"accuracy {f(am,'.2f')} ± {f(as_,'.2f')}%. **Energy is missing for "
            f"these runs** (orchestrator was not capturing jtop power at the time). "
            f"Re-run the sweep with the current orchestrator to collect energy."
        )
    else:
        para = (
            f"We re-ran {args.model} on {args.dataset} ({scope_str} — the same "
            f"sample count used for the other devices in the paper) {len(rows)} "
            f"times on the Jetson AGX Orin (MAXN power mode) "
            f"with a {args.cooldown // 60}-minute idle cooldown between runs. "
            f"Per-run total energy (sum of all -IN_POWER rails: VDD_GPU_SOC, "
            f"VDD_CPU_CV, VIN_SYS_5V0, VDDQ_VDD2_1V8AO; integrated as the same "
            f"analysis script used for the other devices) was "
            f"{f(tem,'.2f')} ± {f(tes,'.2f')} Wh (RSD {f(ter,'.2f')}%). "
            f"Wall-clock time was {f(tm,'.1f')} ± {f(ts_,'.1f')} s "
            f"(RSD {f(tr,'.2f')}%), average total module power "
            f"{f(pm,'.2f')} ± {f(ps_,'.2f')} W, and benchmark accuracy "
            f"{f(am,'.2f')} ± {f(as_,'.2f')}%. Power was sampled at 1 Hz via jtop "
            f"(jetson-stats), reading the same per-rail registers the Jetson "
            f"Power GUI exposes; the integration method (sum of per-sample power "
            f"divided by 3600) is identical to the paper's original analysis "
            f"code, so figures are directly comparable."
        )
    lines.append(para)

    text = "\n".join(lines) + "\n"
    (results_dir / "summary_stats.txt").write_text(text)
    print("\n" + text)


def _write_plot(rows, results_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping variance_plot.png)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [f"Run {r.get('run_id')}" for r in rows]
    for ax, key, ylabel in [(axes[0], "energy_total_wh", "Total Energy (Wh, all rails)"),
                            (axes[1], "duration_s", "Time (s)")]:
        vals = [r.get(key) or 0.0 for r in rows]
        ax.bar(labels, vals, color="#3a7bd5", alpha=0.8)
        if len(vals) > 1:
            m, sd, _ = _msd(vals)
            if m is not None:
                ax.axhline(m, color="red", linestyle="--", linewidth=1,
                           label=f"mean={m:.2f}  std={sd:.2f}")
                ax.legend(loc="upper right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Multi-run variance ({args_model_str_for_plot}, AGX Orin, n={len(rows)})")
    plt.tight_layout()
    plt.savefig(results_dir / "variance_plot.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {results_dir / 'variance_plot.png'}")


# ---------- aggregate across (model, benchmark) sweeps ----------

def _aggregate(scan_root: Path):
    """Walk scan_root for any run_*/run_meta.json files. Group by
    (model, dataset). Write per-metric pivot tables and a long-format CSV.

    Output structure:
      <scan_root>/_aggregate/
        aggregate_metrics.csv    long-format: model,benchmark,n_runs,metric,mean,std,rsd_pct
        accuracy_pct.csv         pivot:  rows=models, cols=benchmarks
        energy_wh.csv            pivot:  total energy (paper headline)
        energy_compute_wh.csv    pivot:  GPU_SOC + CPU_CV only
        time_s.csv               pivot:  wall-clock seconds
        avg_power_w.csv          pivot:  average total power
        aggregate_table.txt      human-readable, paste-ready
    """
    from collections import defaultdict
    if not scan_root.exists():
        print(f"ERROR: scan root not found: {scan_root}")
        sys.exit(2)

    metas_by_group = defaultdict(list)
    for meta_path in scan_root.rglob("run_meta.json"):
        # only count run_*/run_meta.json (skip stray meta files)
        if not meta_path.parent.name.startswith("run_"):
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        model = meta.get("model")
        dataset = meta.get("dataset")
        if model and dataset:
            metas_by_group[(model, dataset)].append(meta)

    if not metas_by_group:
        print(f"No (model, dataset) groups found under {scan_root}")
        return

    print(f"Found {len(metas_by_group)} (model, dataset) groups across {scan_root}:")
    for (model, dataset), metas in sorted(metas_by_group.items()):
        print(f"  {model:<25} {dataset:<20} n_runs={len(metas)}")

    # Build long-format rows
    METRICS = [
        ("accuracy_pct",      "accuracy_pct"),
        ("duration_s",        "time_s"),
        ("energy_wh",         "energy_wh"),          # = total (headline)
        ("avg_power_w",       "avg_power_w"),        # = total
        ("energy_compute_wh", "energy_compute_wh"),
        ("energy_memory_wh",  "energy_memory_wh"),
        ("energy_system_wh",  "energy_system_wh"),
    ]
    long_rows = []
    for (model, dataset), metas in sorted(metas_by_group.items()):
        for src_key, label in METRICS:
            vals = [m.get(src_key) for m in metas if m.get(src_key) is not None]
            if vals:
                mean, sd, rsd = _msd(vals)
            else:
                mean, sd, rsd = None, None, None
            long_rows.append({
                "model": model, "benchmark": dataset, "n_runs": len(vals),
                "metric": label, "mean": mean, "std": sd, "rsd_pct": rsd,
            })

    out_dir = scan_root / "_aggregate"
    out_dir.mkdir(exist_ok=True)

    # Long-format CSV
    long_csv = out_dir / "aggregate_metrics.csv"
    with open(long_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "benchmark", "n_runs",
                                          "metric", "mean", "std", "rsd_pct"])
        w.writeheader()
        for r in long_rows:
            w.writerow(r)
    print(f"\n  wrote {long_csv}")

    # Per-metric pivot CSVs
    PIVOT_METRICS = ["accuracy_pct", "energy_wh", "energy_compute_wh",
                     "time_s", "avg_power_w"]
    for metric in PIVOT_METRICS:
        _write_pivot_csv(long_rows, metric, out_dir / f"{metric}.csv")

    # Human-readable table
    _write_aggregate_text(long_rows, out_dir / "aggregate_table.txt")


def _write_pivot_csv(long_rows, metric, out_path: Path):
    """Pivot: rows = models, cols = benchmarks, cells = 'mean ± std'."""
    relevant = [r for r in long_rows if r["metric"] == metric]
    if not relevant:
        return
    models = sorted({r["model"] for r in relevant})
    benchmarks = sorted({r["benchmark"] for r in relevant})
    cells = {(r["model"], r["benchmark"]): r for r in relevant}
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + benchmarks)
        for model in models:
            row = [model]
            for bench in benchmarks:
                c = cells.get((model, bench))
                if c and c["mean"] is not None:
                    row.append(f"{c['mean']:.4f} ± {c['std']:.4f}")
                else:
                    row.append("N/A")
            w.writerow(row)
    print(f"  wrote {out_path}")


def _write_aggregate_text(long_rows, out_path: Path):
    sections = [
        ("accuracy_pct",      "Accuracy (%)",          ".2f"),
        ("energy_wh",         "Total Energy (Wh)",     ".3f"),
        ("energy_compute_wh", "Compute Energy (Wh)",   ".3f"),
        ("time_s",            "Time (s)",              ".1f"),
        ("avg_power_w",       "Avg Total Power (W)",   ".2f"),
    ]

    def fmt_cell(c, spec):
        if not c or c["mean"] is None:
            return "N/A"
        return f"{c['mean']:{spec}}±{c['std']:{spec}}"

    out_lines = []
    for metric, title, spec in sections:
        relevant = [r for r in long_rows if r["metric"] == metric]
        if not relevant:
            continue
        models = sorted({r["model"] for r in relevant})
        benchmarks = sorted({r["benchmark"] for r in relevant})
        cells = {(r["model"], r["benchmark"]): r for r in relevant}

        # Compute column widths
        cell_strs = {(m, b): fmt_cell(cells.get((m, b)), spec)
                     for m in models for b in benchmarks}
        model_w = max(8, max(len(m) for m in models)) + 2
        col_w = max(
            max(len(b) for b in benchmarks),
            max(len(s) for s in cell_strs.values()),
        ) + 2

        out_lines.append(f"=== {title}  (mean ± std) ===")
        header = f"{'model':<{model_w}}" + "".join(f"{b:>{col_w}}" for b in benchmarks)
        out_lines.append(header)
        out_lines.append("-" * len(header))
        for model in models:
            row = f"{model:<{model_w}}"
            for bench in benchmarks:
                row += f"{cell_strs[(model, bench)]:>{col_w}}"
            out_lines.append(row)
        out_lines.append("")

    text = "\n".join(out_lines) + "\n"
    out_path.write_text(text)
    print()
    print(text)


# ---------- resummarize ----------

def _resummarize(results_dir: Path, args):
    """Re-scan every run_*/ under results_dir, re-parse accuracy from
    whichever scores file exists (json or xlsx), and rewrite summary.csv
    / summary_stats.txt / variance_plot.png. Does not re-run inference."""
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        sys.exit(2)

    rows = []
    for run_dir in sorted(results_dir.glob("run_*")):
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            print(f"  skip {run_dir.name}: no run_meta.json")
            continue
        meta = json.loads(meta_path.read_text())
        outputs_dir = run_dir / "outputs"
        # Re-find scores file (any format) and re-parse accuracy
        sf = _find_scores_file(outputs_dir)
        accuracy = _parse_accuracy(sf) if sf else None
        if accuracy is not None:
            meta["accuracy"] = accuracy
            meta["accuracy_pct"] = 100.0 * accuracy
            meta["scores_file"] = str(sf)
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
            (run_dir / "done.marker").touch()
        print(f"  {run_dir.name}: scores_file={sf.name if sf else None} acc={accuracy}")
        rows.append(meta)

    if not rows:
        print("No runs found.")
        return

    _write_summary_csv(rows, results_dir / "summary.csv")
    _write_stats(rows, results_dir, args)
    _write_plot(rows, results_dir)


# ---------- main ----------

# Mutable global used by _write_plot's suptitle. Set in main() before calling.
args_model_str_for_plot = "Moondream2 on MMBench"


def main():
    global args_model_str_for_plot
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--percentage", type=float, default=DEFAULT_PERCENTAGE,
                    help="Dataset percentage 5-100; CustomVLMBenchmark uses fixed "
                         "random_state=42 so subsets are reproducible. Defaults to "
                         "12 (~519 MMBench samples — matches the count used on "
                         "other devices). Use --n-samples instead for exact counts.")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Exact number of samples (overrides --percentage). The "
                         "subprocess probes the dataset's actual size and computes "
                         "the float percentage that yields exactly N samples. "
                         "Recommended: 519 (MMBench_DEV_EN), 711 (SEEDBench_IMG), "
                         "510 (MMStar), 522 (MME).")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN_S,
                    help="Seconds of idle wait between runs (0 to disable).")
    ap.add_argument("--results-dir", default="results/multirun_agx_orin")
    ap.add_argument("--only-run", type=int, default=None,
                    help="Run just this run index (1-based), useful for retrying.")
    ap.add_argument("--force-no-maxn", action="store_true",
                    help="Proceed even if nvpmodel -q does not show MAXN.")
    ap.add_argument("--resummarize", action="store_true",
                    help="Skip running benchmarks. For each existing run_*/run_meta.json "
                         "in --results-dir, re-find the scores file (json or xlsx), "
                         "re-parse accuracy, and rewrite summary.csv / summary_stats.txt / "
                         "variance_plot.png. Useful for recovering data after orchestrator "
                         "bug fixes without re-running the (expensive) inference.")
    ap.add_argument("--aggregate", nargs="?", const="results", default=None,
                    metavar="SCAN_ROOT",
                    help="Walk SCAN_ROOT (default 'results') for any sweep results "
                         "(subdirs containing run_*/run_meta.json), group by "
                         "(model, dataset), and write pivot tables to "
                         "SCAN_ROOT/_aggregate/. Use after all sweeps complete to "
                         "get cross-model cross-benchmark accuracy/energy/time tables. "
                         "Skips inference.")
    args = ap.parse_args()
    args_model_str_for_plot = f"{args.model} on {args.dataset}"

    results_dir = Path(args.results_dir)

    if args.aggregate is not None:
        _aggregate(Path(args.aggregate))
        return

    if args.resummarize:
        _resummarize(results_dir, args)
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    is_maxn, modeline = verify_maxn()
    print(f"[nvpmodel -q] {modeline}")
    if not is_maxn and not args.force_no_maxn:
        print("ERROR: Power mode is not MAXN. Set MAXN with `sudo nvpmodel -m 0` "
              "(may require reboot) or rerun with --force-no-maxn.")
        sys.exit(2)

    indices = [args.only_run] if args.only_run else list(range(1, args.runs + 1))
    completed_rows = []
    for i in indices:
        run_dir = results_dir / f"run_{i}"
        marker = run_dir / "done.marker"
        if marker.exists():
            print(f"\n[run {i}] already complete (delete {marker} to re-run) — loading meta")
            completed_rows.append(json.loads((run_dir / "run_meta.json").read_text()))
            continue
        if i > 1 and args.cooldown > 0:
            cooldown(args.cooldown)
        completed_rows.append(run_one(i, args.runs, run_dir, args))

    # Summarize whatever we have. Energy/time variance is the reviewer's
    # primary ask, so we write summary.csv as long as ANY run produced timing
    # data — even if accuracy parsing failed for all of them.
    with_timing = [r for r in completed_rows if r.get("duration_s")]
    with_acc = [r for r in completed_rows if r.get("accuracy") is not None]
    print(f"\nRuns complete: {len(with_timing)}/{len(completed_rows)} have timing data,"
          f" {len(with_acc)} also have accuracy.")

    if not with_timing:
        print("No timing data captured; skipping summary.")
        return

    _write_summary_csv(completed_rows, results_dir / "summary.csv")
    _write_stats(with_timing, results_dir, args)
    _write_plot(with_timing, results_dir)


if __name__ == "__main__":
    main()
