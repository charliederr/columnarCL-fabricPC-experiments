#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
run_script="${repo_root}/scripts/run_codex_cifar10_depth_spanning.sh"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"

seed="${SEED:-42}"
lr="${LR:-0.005}"
num_epochs="${NUM_EPOCHS:-20}"
diagnose_mode="${DIAGNOSE_MODE:-nodiag}"
post_training_diagnostics="${POST_TRAINING_DIAGNOSTICS:-core}"
weights="${WEIGHTS:-0.0 0.00025 0.0005 0.001}"
support_mask="${SUPPORT_MASK:-1,1,1,1,1,1,1,1,1,1}"
master_log="${repo_root}/results/codex_inward_shell_promotion_10col3shared_seed${seed}_${hostname_value}_${timestamp}.log"

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
    echo "post_training_diagnostics: $post_training_diagnostics"
    echo "weights: $weights"
    echo "support_mask: $support_mask"
    echo "num_columns: 10"
    echo "num_shared: 3"
    echo "active_nonshared: 7"
    echo "column_mode: all_active"
    echo "column_grid: stage4"
    echo "embed_dim: 64"
    echo "microcolumn_dim: 32"
    echo "column_gaussian_energy_mode: sum"
    echo "column_gaussian_precision: 1.0"
    echo "column_gaussian_reference_sites: 16"
    echo "combiner: shell_attention"
    echo "outer_shell_context: on"
    echo "column_shell_bridge: on"
    echo "column_shell_readout: off"
    echo "outer_shell_context_to_bridge: off"
    echo "outer_shell_context_bridge_scale: 0.0"
    echo "outer_shell_context_shell_prediction_weight: 0.0"
    echo "shell_lr_multipliers: 1,1.5,2,3"
    echo "readout_mode: nobypass"
    echo "column_teacher_weight: 0.0"
    echo "shell_teacher_weights: 0,0,0,0"
    echo "column_shell_teacher_weights: 0,0,0,0"
    echo "outer_shell_context_teacher_weight: 0.0"
    echo "outer_shell_context_evidence: off"
    echo "outer_shell_context_evidence_teacher_weight: 0.0"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

    for weight in $weights; do
        echo
        echo "======================================================================"
        echo "Starting inward_shell_promotion_weight=${weight} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
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
            "0.0" \
            "all_active" \
            "$support_mask" \
            "$post_training_diagnostics" \
            "$weight"

        echo "Finished inward_shell_promotion_weight=${weight} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    done

    echo
    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
