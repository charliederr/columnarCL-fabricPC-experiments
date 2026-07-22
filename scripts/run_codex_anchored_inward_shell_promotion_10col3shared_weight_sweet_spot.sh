#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"

seeds="${SEEDS:-42}"
weights="${WEIGHTS:-0.0001 0.000375 0.00075 0.001}"
promotion_pairs="${PROMOTION_PAIRS:-outer_to_middle,middle_to_inner}"
target_gradient_scale="${INWARD_SHELL_PROMOTION_TARGET_GRADIENT_SCALE:-0.0}"
warmup_epochs="${INWARD_SHELL_PROMOTION_WARMUP_EPOCHS:-0.0}"
ramp_epochs="${INWARD_SHELL_PROMOTION_RAMP_EPOCHS:-0.0}"
promotion_pairs_label="${promotion_pairs//,/_}"
target_gradient_scale_label="${target_gradient_scale//./p}"
warmup_label="${warmup_epochs//./p}"
ramp_label="${ramp_epochs//./p}"
weights_label="${weights// /_}"
weights_label="${weights_label//./p}"
master_log="${repo_root}/results/codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot_pairs${promotion_pairs_label}_weights${weights_label}_iptg${target_gradient_scale_label}_ipw${warmup_label}_ipr${ramp_label}_${hostname_value}_${timestamp}.log"

cd "$repo_root"
mkdir -p results

{
    echo "repo: $repo_root"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --short --branch
    echo "seeds: $seeds"
    echo "weights: $weights"
    echo "promotion_pairs: $promotion_pairs"
    echo "inward_shell_promotion_target_gradient_scale: $target_gradient_scale"
    echo "inward_shell_promotion_warmup_epochs: $warmup_epochs"
    echo "inward_shell_promotion_ramp_epochs: $ramp_epochs"
    echo "lr: ${LR:-0.005}"
    echo "num_epochs: ${NUM_EPOCHS:-20}"
    echo "diagnose_mode: ${DIAGNOSE_MODE:-nodiag}"
    echo "post_training_diagnostics: ${POST_TRAINING_DIAGNOSTICS:-core}"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

    for seed in $seeds; do
        echo
        echo "======================================================================"
        echo "Starting seed=${seed} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "======================================================================"

        SEED="$seed" \
        WEIGHTS="$weights" \
        PROMOTION_PAIRS="$promotion_pairs" \
        INWARD_SHELL_PROMOTION_TARGET_GRADIENT_SCALE="$target_gradient_scale" \
        INWARD_SHELL_PROMOTION_WARMUP_EPOCHS="$warmup_epochs" \
        INWARD_SHELL_PROMOTION_RAMP_EPOCHS="$ramp_epochs" \
        bash scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh

        echo "Finished seed=${seed} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    done

    echo
    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
