#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
run_script="${repo_root}/scripts/run_codex_cifar10_depth_spanning.sh"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"

lr="${LR:-0.005}"
num_epochs="${NUM_EPOCHS:-20}"
diagnose_mode="${DIAGNOSE_MODE:-composer_shells}"
seeds_text="${SEEDS:-42}"
column_mode="${COLUMN_MODE:-first_sparse}"

if [[ "$column_mode" != "first_sparse" && "$column_mode" != "random_sparse" ]]; then
    echo "COLUMN_MODE must be 'first_sparse' or 'random_sparse' for this sweep." >&2
    exit 2
fi

if [[ "$#" -gt 0 ]]; then
    active_nonshared_text="$*"
else
    active_nonshared_text="${ACTIVE_NONSHARED_VALUES:-1 3 5}"
fi

seed_label="${seeds_text// /_}"
support_label="${active_nonshared_text// /_}"
master_log="${repo_root}/results/codex_support_sparsity_10col3shared_${column_mode}_seeds${seed_label}_active${support_label}_${hostname_value}_${timestamp}.log"

cd "$repo_root"
mkdir -p results

{
    echo "repo: $repo_root"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --short --branch
    echo "lr: $lr"
    echo "num_epochs: $num_epochs"
    echo "diagnose_mode: $diagnose_mode"
    echo "seeds: $seeds_text"
    echo "num_columns: 10"
    echo "num_shared: 3"
    echo "active_nonshared_values: $active_nonshared_text"
    echo "column_mode: $column_mode"
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

    for seed in $seeds_text; do
        for active_nonshared in $active_nonshared_text; do
            total_active=$((3 + active_nonshared))
            echo
            echo "======================================================================"
            echo "Starting seed=${seed} active_nonshared=${active_nonshared} total_active_columns=${total_active} column_mode=${column_mode} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
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
                "$active_nonshared" \
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
                "$column_mode"

            echo "Finished seed=${seed} active_nonshared=${active_nonshared} total_active_columns=${total_active} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
        done
    done

    echo
    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
