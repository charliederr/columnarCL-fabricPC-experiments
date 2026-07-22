#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
export PROMOTION_PAIRS="${PROMOTION_PAIRS:-outer_to_middle,middle_to_inner}"
export WEIGHTS="${WEIGHTS:-0.00025 0.0005}"
export INWARD_SHELL_PROMOTION_TARGET_GRADIENT_SCALE="${INWARD_SHELL_PROMOTION_TARGET_GRADIENT_SCALE:-0.0}"
export INWARD_SHELL_PROMOTION_WARMUP_EPOCHS="${INWARD_SHELL_PROMOTION_WARMUP_EPOCHS:-0.0}"
export INWARD_SHELL_PROMOTION_RAMP_EPOCHS="${INWARD_SHELL_PROMOTION_RAMP_EPOCHS:-0.0}"

cd "$repo_root"
bash scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh
