#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
runner="${repo_root}/scripts/run_codex_cifar10_depth_spanning.sh"
seed="${SEED:-42}"
lr="${LR:-0.005}"
diagnose_mode="${DIAGNOSE_MODE:-shells}"
num_epochs="${NUM_EPOCHS:-20}"
weights="${WEIGHTS:-0.0 0.025 0.075 0.125}"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="${repo_root}/results/codex_teacher_weight_sweep_seed${seed}_${hostname_value}_${timestamp}.log"

cd "$repo_root"
mkdir -p results

{
    echo "repo: $repo_root"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --short --branch
    echo "seed: $seed"
    echo "lr: $lr"
    echo "num_epochs: $num_epochs"
    echo "diagnose_mode: $diagnose_mode"
    echo "weights: $weights"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo

    for weight in $weights; do
        echo "======================================================================"
        echo "Starting column_teacher_weight=$weight at $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "======================================================================"
        bash "$runner" "$seed" "$lr" "$diagnose_mode" "$num_epochs" "$weight"
        echo "Finished column_teacher_weight=$weight at $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo
    done

    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
