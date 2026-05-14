#!/usr/bin/env bash
# Launcher for the multi-run reproducibility experiment on Jetson AGX Orin.
# See MULTIRUN_README.md for full usage.
#
# Examples:
#   ./run_multirun_agx_orin.sh                       # 3 runs of Moondream2 on full MMBench
#   ./run_multirun_agx_orin.sh --percentage 25       # 3 runs on 25% sample (~30 min total)
#   ./run_multirun_agx_orin.sh --only-run 2          # resume / redo just run 2
#   ./run_multirun_agx_orin.sh --runs 5              # 5 repetitions instead of 3
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Sanity: jtop (for live resource logging — CPU/RAM/GPU util/freq).
# Power and temperature come from the Jetson Power GUI, not jtop.
if ! python3 -c "from jtop import jtop" 2>/dev/null; then
    echo "ERROR: jtop (jetson-stats) is not installed or not importable." >&2
    echo "       Install: sudo pip install -U jetson-stats" >&2
    echo "       Then ensure the service is running:" >&2
    echo "         sudo systemctl enable --now jetson_stats.service" >&2
    exit 1
fi

# Sanity: torch + CUDA. If broken, point at SETUP_NOTES.md.
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: torch.cuda.is_available() is False." >&2
    echo "       The JetPack PyTorch wheel may have been clobbered by a pip install." >&2
    echo "       See SETUP_NOTES.md for the one-line restore command." >&2
    exit 1
fi

exec python3 multirun_agx_orin.py "$@"
