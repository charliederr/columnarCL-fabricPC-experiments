#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"

seeds="${SEEDS:-99 7}"
weights="${WEIGHTS:-0.0005}"
promotion_pairs="${PROMOTION_PAIRS:-outer_to_middle,middle_to_inner}"
target_gradient_scale="${INWARD_SHELL_PROMOTION_TARGET_GRADIENT_SCALE:-0.0}"
promotion_pairs_label="${promotion_pairs//,/_}"
target_gradient_scale_label="${target_gradient_scale//./p}"
master_log="${repo_root}/results/codex_anchored_inward_shell_promotion_10col3shared_replicate_pairs${promotion_pairs_label}_iptg${target_gradient_scale_label}_${hostname_value}_${timestamp}.log"

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
        bash scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh

        echo "Finished seed=${seed} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    done

    echo
    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
