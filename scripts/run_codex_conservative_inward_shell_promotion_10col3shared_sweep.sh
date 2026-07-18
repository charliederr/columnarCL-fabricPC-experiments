#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
export PROMOTION_PAIRS="${PROMOTION_PAIRS:-outer_to_middle,middle_to_inner}"
export WEIGHTS="${WEIGHTS:-0.0001 0.00025}"

cd "$repo_root"
bash scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh
