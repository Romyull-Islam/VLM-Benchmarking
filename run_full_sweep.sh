#!/usr/bin/env bash
# run_full_sweep.sh — launch every (model, benchmark) sweep in sequence.
#
# Each sweep = 3 cold-start runs of one model on one benchmark with 5-min
# cooldowns between runs. Energy/temp/CPU/RAM/GPU captured live via jtop;
# accuracy parsed from VLMEvalKit's scores file.
#
# Resumable: a sweep is skipped if every run_*/done.marker already exists
# under its results/ directory. To force a re-run of one sweep, rm its dir.
#
# Usage:
#   ./run_full_sweep.sh                    # all models × all benchmarks
#   MODELS="Moondream2 SmolVLM2-256M" ./run_full_sweep.sh   # subset
#   FROM_SCRATCH=1 ./run_full_sweep.sh     # rm -rf results/ first (fresh)

set -uo pipefail   # NOT -e: one failed sweep should not stop the rest
cd "$(dirname "$(readlink -f "$0")")"

# ------------- configuration -------------

# Model list (override via MODELS env var). Default = the 3 models in the
# paper's AGX Orin table. To include more (h2ovl, InternVL2_5, etc.), set:
#   MODELS="SmolVLM2-256M SmolVLM2-500M Moondream2 h2ovl-mississippi-1b ..."
MODELS=${MODELS:-"SmolVLM2-256M SmolVLM2-500M Moondream2"}

# (benchmark, percentage) pairs — exact percentages that produce your
# target sample counts on the actual VLMEvalKit dataset sizes:
#   MMBench_DEV_EN  12% → 519
#   SEEDBench_IMG    5% → 711
#   MMStar          34% → 510
#   MME             22% → 522
BENCH_PCTS=(
  "MMBench_DEV_EN:12"
  "SEEDBench_IMG:5"
  "MMStar:34"
  "MME:22"
)

RUNS=${RUNS:-3}
COOLDOWN=${COOLDOWN:-300}

# ------------- pre-flight -------------

if [ "${FROM_SCRATCH:-0}" = "1" ]; then
    echo ">>> FROM_SCRATCH=1 -> wiping results/ before sweep"
    rm -rf results
fi

mkdir -p results
LOGFILE="results/sweep_$(date +%Y%m%d_%H%M%S).log"
echo ">>> log: $LOGFILE"
echo "models: $MODELS" | tee -a "$LOGFILE"
echo "benchmarks: ${BENCH_PCTS[*]}" | tee -a "$LOGFILE"
echo "runs/sweep: $RUNS  cooldown_s: $COOLDOWN" | tee -a "$LOGFILE"

n_total=0
n_done=0
n_failed=0
sweep_idx=0
total_sweeps=$(( $(echo "$MODELS" | wc -w) * ${#BENCH_PCTS[@]} ))

# ------------- sweep loop -------------

for model in $MODELS; do
    for entry in "${BENCH_PCTS[@]}"; do
        bench="${entry%%:*}"
        pct="${entry##*:}"
        sweep_idx=$((sweep_idx + 1))
        safe_model="${model//\//_}"
        safe_bench="${bench//\//_}"
        results_dir="results/${safe_model}__${safe_bench}"

        echo ""
        echo "================================================================"
        echo "  [${sweep_idx}/${total_sweeps}]  model=${model}  bench=${bench}  pct=${pct}%  -> ${results_dir}"
        echo "================================================================"

        sweep_log="${results_dir}.log"
        mkdir -p "$(dirname "$sweep_log")"

        ./run_multirun_agx_orin.sh \
            --model "$model" \
            --dataset "$bench" \
            --percentage "$pct" \
            --runs "$RUNS" \
            --cooldown "$COOLDOWN" \
            --results-dir "$results_dir" \
            2>&1 | tee "$sweep_log"
        rc=${PIPESTATUS[0]}
        n_total=$((n_total + 1))
        if [ $rc -eq 0 ]; then
            n_done=$((n_done + 1))
            echo ">>> sweep ${sweep_idx} OK"
        else
            n_failed=$((n_failed + 1))
            echo ">>> sweep ${sweep_idx} FAILED (rc=$rc) — continuing"
        fi
    done
done

# ------------- aggregate -------------

echo ""
echo "================================================================"
echo "  SWEEPS DONE: ${n_done}/${n_total} OK, ${n_failed} failed"
echo "  Aggregating across all (model, benchmark) sweeps..."
echo "================================================================"

./run_multirun_agx_orin.sh --aggregate results 2>&1 | tee -a "$LOGFILE"

echo ""
echo ">>> Full sweep done."
echo ">>> Cross-model cross-benchmark tables: results/_aggregate/"
echo "    - aggregate_table.txt   (paste-ready)"
echo "    - accuracy_pct.csv      (model x benchmark)"
echo "    - energy_wh.csv         (total energy, paper convention)"
echo "    - time_s.csv            (wall-clock)"
echo "    - aggregate_metrics.csv (long-format for pandas)"
echo ">>> Per-sweep raw data: results/<model>__<benchmark>/"
echo "    - run_N/jtop_log.csv (1Hz timeseries: power, temps, CPU, RAM, GPU util/freq)"
echo "    - run_N/run_meta.json (start/end/duration/accuracy/energy/temps)"
echo "    - summary.csv, summary_stats.txt"
