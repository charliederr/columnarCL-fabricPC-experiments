#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
python_bin="${FPC_PYTHON:-/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python}"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="${repo_root}/results/codex_shell_bridge_readout_column_capacity_sweep_${hostname_value}_${timestamp}.log"

cd "$repo_root"
mkdir -p results

run_case() {
    local case_name="$1"
    local seed="$2"
    local num_columns="$3"
    local num_shared="$4"
    local active_nonshared="$5"
    local columns_label="${num_columns}col"
    local log_path="${repo_root}/results/codex_resnet18_${columns_label}_colshellreadouton_colshellbridgeon_zero_teacher_seed${seed}_lr0p005_ep20_shells_${hostname_value}_$(date +%Y%m%d_%H%M%S).log"

    echo "======================================================================"
    echo "Starting case=$case_name at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "seed: $seed"
    echo "num_columns: $num_columns"
    echo "num_shared: $num_shared"
    echo "active_nonshared: $active_nonshared"
    echo "lr: 0.005"
    echo "num_epochs: 20"
    echo "column_teacher_weight: 0.0"
    echo "shell_teacher_weights: 0,0,0,0"
    echo "column_shell_teacher_weights: 0,0,0,0"
    echo "column_shell_readout: on"
    echo "column_shell_bridge: on"
    echo "readout_mode: nobypass"
    echo "log_path: $log_path"
    echo "======================================================================"

    {
        echo "repo: $repo_root"
        echo "git_commit: $(git rev-parse HEAD)"
        echo "git_status:"
        git status --short --branch
        echo "python: $python_bin"
        echo "case: $case_name"
        echo "seed: $seed"
        echo "num_columns: $num_columns"
        echo "num_shared: $num_shared"
        echo "active_nonshared: $active_nonshared"
        "$python_bin" -c "import jax; print(jax.devices()); print(jax.default_backend())"
        "$python_bin" scripts/train_cifar10_depth_spanning.py \
            --model resnet18 \
            --activation leaky_relu \
            --column_activation leaky_relu \
            --num_columns "$num_columns" \
            --num_shared "$num_shared" \
            --active_nonshared "$active_nonshared" \
            --column_mode all_active \
            --combiner sum \
            --embed_dim 64 \
            --microcolumn_dim 32 \
            --batch_size 128 \
            --num_epochs 20 \
            --lr 0.005 \
            --weight_decay 0.01 \
            --infer_steps 40 \
            --eta_infer 0.1 \
            --infer_max_norm 1.0 \
            --eval_every 1 \
            --seed "$seed" \
            --layer_norm_tokens \
            --fix_ln_gamma \
            --column_teacher_weight 0.0 \
            --shell_teacher_weights 0,0,0,0 \
            --column_shell_teacher_weights 0,0,0,0 \
            --column_shell_readout \
            --column_shell_bridge \
            --diagnose_shells
    } 2>&1 | tee "$log_path"

    echo "Finished case=$case_name at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "case_log: $log_path"
    echo
}

{
    echo "repo: $repo_root"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --short --branch
    echo "python: $python_bin"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
    echo "planned_cases:"
    echo "1. seed99_6column_shell_bridge_plus_direct_readout_zero_teacher_energy"
    echo "2. seed42_6column_shell_bridge_plus_direct_readout_zero_teacher_energy"
    echo

    run_case "seed99_6column_shell_bridge_plus_direct_readout_zero_teacher_energy" 99 6 3 3
    run_case "seed42_6column_shell_bridge_plus_direct_readout_zero_teacher_energy" 42 6 3 3

    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
