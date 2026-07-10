#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
python_bin="${FPC_PYTHON:-/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python}"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
seeds="${SEEDS:-42}"
weights="${WEIGHTS:-0.0005}"
lr="${LR:-0.005}"
num_epochs="${NUM_EPOCHS:-20}"
diagnose_mode="${DIAGNOSE_MODE:-composer_shells}"
shell_lr_multipliers="${SHELL_LR_MULTIPLIERS:-1,1.5,2,3}"
master_log="${repo_root}/results/codex_shell_context_prediction_with_readout_10col3shared_${hostname_value}_${timestamp}.log"

cd "$repo_root"
mkdir -p results

{
    echo "repo: $repo_root"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --short --branch
    echo "python: $python_bin"
    echo "seeds: $seeds"
    echo "weights: $weights"
    echo "lr: $lr"
    echo "num_epochs: $num_epochs"
    echo "diagnose_mode: $diagnose_mode"
    echo "shell_lr_multipliers: $shell_lr_multipliers"
    echo "architecture: 10 columns, 3 shared, 7 active non-shared"
    echo "combiner: shell_attention"
    echo "outer_shell_context: on"
    echo "outer_shell_context direct readout: retained"
    echo "outer_shell_context shell prediction: added local objective"
    echo "outer_shell_context_evidence: off"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo

    for weight in $weights; do
        for seed in $seeds; do
            echo "======================================================================"
            echo "Starting seed=$seed shell_prediction_weight=$weight at $(date '+%Y-%m-%d %H:%M:%S %Z')"
            echo "======================================================================"
            bash "$repo_root/scripts/run_codex_cifar10_depth_spanning.sh" \
                "$seed" \
                "$lr" \
                "$diagnose_mode" \
                "$num_epochs" \
                0.0 \
                0,0,0,0 \
                nobypass \
                0,0,0,0 \
                off \
                off \
                shell_attention \
                on \
                10 \
                3 \
                7 \
                "$shell_lr_multipliers" \
                0.0 \
                off \
                0.0 \
                "$weight"
            echo "Finished seed=$seed shell_prediction_weight=$weight at $(date '+%Y-%m-%d %H:%M:%S %Z')"
            echo
        done
    done

    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
} 2>&1 | tee "$master_log"

echo "master_log: $master_log"
