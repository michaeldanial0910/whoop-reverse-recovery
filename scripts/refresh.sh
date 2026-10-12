#!/usr/bin/env bash
# Pull new WHOOP data and rebuild everything the dashboard reads.
# The pre-registered irregularity test (m3_irregularity.py) is deliberately NOT re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/sync.py            # new cycles -> data/raw, then normalize_data.py + features.py
python src/m3_anomalies.py    # out-of-fold residuals and surprise flags
python src/m3_archetypes.py   # kinds of days
echo "Done. Reload the dashboard."
