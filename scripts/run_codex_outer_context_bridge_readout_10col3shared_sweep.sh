#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
run_script="${repo_root}/scripts/run_codex_cifar10_depth_spanning.sh"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"

seed="${SEED:-42}"
lr="${LR:-0.005}"
num_epochs="${NUM_EPOCHS:-20}"
diagnose_mode="${DIAGNOSE_MODE:-composer_shells}"

master_log="${repo_root}/results/codex_outer_context_bridge_readout_10col3shared_${hostname_value}_${timestamp}.log"

case_names=(
    "bridge_only"
    "readout_only"
    "bridge_plus_readout"
)
column_shell_readout_modes=(
    "off"
    "on"
    "on"
)
column_shell_bridge_modes=(
    "on"
    "off"
    "on"
)

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
    echo "num_columns: 10"
    echo "num_shared: 3"
    echo "active_nonshared: 7"
    echo "combiner: shell_attention"
    echo "outer_shell_context: on"
    echo "shell_lr_multipliers: 1,1.5,2,3"
    echo "readout_mode: nobypass"
    echo "column_teacher_weight: 0.0"
    echo "shell_teacher_weights: 0,0,0,0"
    echo "column_shell_teacher_weights: 0,0,0,0"
    echo "outer_shell_context_teacher_weight: 0.0"
    echo "outer_shell_context_evidence: off"
    echo "outer_shell_context_evidence_teacher_weight: 0.0"
    echo "outer_shell_context_shell_prediction_weight: 0.0"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

    for idx in "${!case_names[@]}"; do
        case_name="${case_names[$idx]}"
        column_shell_readout_mode="${column_shell_readout_modes[$idx]}"
        column_shell_bridge_mode="${column_shell_bridge_modes[$idx]}"

        echo
        echo "======================================================================"
        echo "Starting case=${case_name} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "column_shell_readout: ${column_shell_readout_mode}"
        echo "column_shell_bridge: ${column_shell_bridge_mode}"
        echo "======================================================================"

        bash "$run_script" \
            "$seed" \
            "$lr" \
            "$diagnose_mode" \
            "$num_epochs" \
            "0.0" \
            "0,0,0,0" \
            "nobypass" \
            "0,0,0,0" \
            "$column_shell_readout_mode" \
            "$column_shell_bridge_mode" \
            "shell_attention" \
            "on" \
            "10" \
            "3" \
            "7" \
            "1,1.5,2,3" \
            "0.0" \
            "off" \
            "0.0" \
            "0.0"

        echo "Finished case=${case_name} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    done

    echo
    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
