"""Smoke test: run Moondream2 on MMBench_DEV_EN at 5% to verify the Jetson stack
end-to-end before launching the full multi-run experiment.

This wraps the existing CustomVLMBenchmark class without modifying it.
Expect: ~3-5 minutes including model download (~4 GB) and dataset fetch.
"""
import logging
from custom_vlm_benchmark import CustomVLMBenchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("smoke")

if __name__ == "__main__":
    bench = CustomVLMBenchmark(dataset_percentage=5)
    result = bench.run_benchmark(model_name="Moondream2", dataset_name="MMBench_DEV_EN")
    if "error" in result:
        log.error("Smoke test FAILED: %s", result["error"])
        raise SystemExit(1)
    log.info("Smoke test PASSED")
    log.info("Output dir: %s", result["output_dir"])
    log.info("Scores file: %s", result["scores_file"])
