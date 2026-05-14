"""One cold-start benchmark run, launched as a subprocess by multirun_agx_orin.py.

Running each repetition in its own process guarantees a fresh model load
(no warm Transformers cache from a previous run) — that's the only reason
this file exists separately from the orchestrator. It otherwise just calls
into the existing CustomVLMBenchmark class.

Writes a small run_meta.json with start/end timestamps and the path to the
scores file (which the orchestrator parses later for accuracy).

Two ways to control sample count:
  --percentage P   : Pass P (float) directly to CustomVLMBenchmark.
                     subset_size = int(total * P/100) — may differ from
                     your target by 1 due to truncation.
  --n-samples N    : Probe the dataset's actual size first, then compute
                     the float percentage that yields exactly N samples
                     (P = (N + 0.5) / total * 100). Overrides --percentage
                     when set. CustomVLMBenchmark requires P >= 5, so very
                     small N on very large datasets will be rejected.
"""
import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

from custom_vlm_benchmark import CustomVLMBenchmark

log = logging.getLogger("multirun_single_run")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _exact_percentage_for_n(dataset_name: str, target_n: int) -> float:
    """Build the dataset (cheap; just reads the TSV from cache) to learn its
    total size, then return a float percentage P such that
    int(total * P/100) == target_n.

    Adding 0.5 to target_n in the numerator guarantees the floor function
    inside CustomVLMBenchmark lands exactly on target_n (instead of
    target_n - 1, which is what the naive computation gives ~half the time).
    """
    from vlmeval.dataset import build_dataset
    ds = build_dataset(dataset_name)
    total = len(ds.data)
    if target_n >= total:
        log.info(f"--n-samples ({target_n}) >= dataset total ({total}); using full dataset")
        return 100.0
    pct = (target_n + 0.5) / total * 100.0
    if pct < 5.0:
        raise ValueError(
            f"--n-samples={target_n} from total={total} requires percentage={pct:.4f}, "
            f"below CustomVLMBenchmark's minimum of 5%. Increase n or run a different "
            f"dataset variant."
        )
    log.info(f"--n-samples mode: target={target_n} of total={total} → percentage={pct:.4f}%")
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Moondream2")
    ap.add_argument("--dataset", default="MMBench_DEV_EN")
    ap.add_argument("--percentage", type=float, default=100.0,
                    help="Float percentage 5-100. Used directly if --n-samples not given.")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Exact number of samples (overrides --percentage).")
    ap.add_argument("--work-dir", required=True,
                    help="Directory passed to CustomVLMBenchmark; outputs land in "
                         "<work-dir>/<model>_<dataset>_<ts>/")
    ap.add_argument("--meta-out", required=True,
                    help="Where to write run_meta.json")
    args = ap.parse_args()

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "n_samples_requested": args.n_samples,
        "start_time": time.time(),
        "start_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        if args.n_samples is not None:
            percentage = _exact_percentage_for_n(args.dataset, args.n_samples)
        else:
            percentage = args.percentage
        meta["percentage_used"] = percentage

        bench = CustomVLMBenchmark(work_dir=args.work_dir, dataset_percentage=percentage)
        result = bench.run_benchmark(model_name=args.model, dataset_name=args.dataset)
        meta["end_time"] = time.time()
        meta["end_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["duration_s"] = meta["end_time"] - meta["start_time"]
        meta["output_dir"] = result.get("output_dir")
        meta["scores_file"] = result.get("scores_file")
        meta["predictions_file"] = result.get("predictions_file")
        meta["error"] = result.get("error")
    except Exception:
        meta["end_time"] = time.time()
        meta["duration_s"] = meta["end_time"] - meta["start_time"]
        meta["error"] = traceback.format_exc()

    Path(args.meta_out).write_text(json.dumps(meta, indent=2, default=str))
    sys.exit(1 if meta.get("error") else 0)


if __name__ == "__main__":
    main()
