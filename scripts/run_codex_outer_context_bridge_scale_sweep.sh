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

scales=("$@")
if [[ "${#scales[@]}" -eq 0 ]]; then
    scales=("0.025" "0.05" "0.10")
fi

master_log="${repo_root}/results/codex_outer_context_bridge_scale_sweep_seed${seed}_${hostname_value}_${timestamp}.log"

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
    echo "column_grid: stage4"
    echo "embed_dim: 64"
    echo "microcolumn_dim: 32"
    echo "column_gaussian_energy_mode: sum"
    echo "column_gaussian_precision: 1.0"
    echo "column_gaussian_reference_sites: 16"
    echo "combiner: shell_attention"
    echo "column_shell_bridge: on"
    echo "outer_shell_context: on"
    echo "outer_shell_context_to_bridge: off"
    echo "shell_lr_multipliers: 1,1.5,2,3"
    echo "readout_mode: nobypass"
    echo "column_teacher_weight: 0.0"
    echo "shell_teacher_weights: 0,0,0,0"
    echo "column_shell_teacher_weights: 0,0,0,0"
    echo "outer_shell_context_teacher_weight: 0.0"
    echo "outer_shell_context_evidence: off"
    echo "outer_shell_context_evidence_teacher_weight: 0.0"
    echo "outer_shell_context_shell_prediction_weight: 0.0"
    echo "outer_shell_context_bridge_scales: ${scales[*]}"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

    for scale in "${scales[@]}"; do
        echo
        echo "======================================================================"
        echo "Starting outer_shell_context_bridge_scale=${scale} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
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
            "off" \
            "on" \
            "shell_attention" \
            "on" \
            "10" \
            "3" \
            "7" \
            "1,1.5,2,3" \
            "0.0" \
            "off" \
            "0.0" \
            "0.0" \
            "off" \
            "stage4" \
            "64" \
            "32" \
            "sum" \
            "1.0" \
            "16" \
            "$scale"

        echo "Finished outer_shell_context_bridge_scale=${scale} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    done

    echo
    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
