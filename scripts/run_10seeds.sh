#!/usr/bin/env bash
# Run the main configuration (config tag P1CI_P6eff050, "direct" selection) for
# the ten outer cross-validation seeds reported in the paper.
#
#   bash scripts/run_10seeds.sh
#
# Set PY to use a specific interpreter, e.g. PY=/path/to/python bash scripts/run_10seeds.sh
set -u
PY="${PY:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

for seed in 40 41 42 456 789 2024 2025 2026 2027 2028; do
  echo "===== seed $seed ====="
  "$PY" src/run_experiment.py P1CI_P6eff050 "$seed" || echo "seed $seed failed"
done
