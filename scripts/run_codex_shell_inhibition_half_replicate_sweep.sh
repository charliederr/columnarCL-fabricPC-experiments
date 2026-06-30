#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
python_bin="${FPC_PYTHON:-/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python}"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="${repo_root}/results/codex_shell_inhibition_half_replicate_sweep_${hostname_value}_${timestamp}.log"
shell_inhibition_strengths="0,0.175,0.11,0.05"
shell_evidence_cascade_scale="0.05,0.05,0.05"

cd "$repo_root"
mkdir -p results

run_case() {
    local case_name="$1"
    local seed="$2"
    local log_path="${repo_root}/results/codex_resnet18_shelldynamicson_halfinhib_column_teacher0p0_shell0_0_0_0_colshell0_0_0_0_colshellreadouton_colshellbridgeon_nobypass_norm_fixedln_seed${seed}_lr0p005_ep20_shells_${hostname_value}_$(date +%Y%m%d_%H%M%S).log"

    echo "======================================================================"
    echo "Starting case=$case_name at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "seed: $seed"
    echo "lr: 0.005"
    echo "num_epochs: 20"
    echo "shell_evidence_cascade: on"
    echo "shell_evidence_cascade_scale: $shell_evidence_cascade_scale"
    echo "shell_inhibition_strengths: $shell_inhibition_strengths"
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
        echo "shell_evidence_cascade_scale: $shell_evidence_cascade_scale"
        echo "shell_inhibition_strengths: $shell_inhibition_strengths"
        "$python_bin" -c "import jax; print(jax.devices()); print(jax.default_backend())"
        "$python_bin" scripts/train_cifar10_depth_spanning.py \
            --model resnet18 \
            --activation leaky_relu \
            --column_activation leaky_relu \
            --num_columns 4 \
            --num_shared 2 \
            --active_nonshared 2 \
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
            --shell_evidence_cascade_scale "$shell_evidence_cascade_scale" \
            --shell_inhibition_strengths "$shell_inhibition_strengths" \
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
    echo "shell_evidence_cascade_scale: $shell_evidence_cascade_scale"
    echo "shell_inhibition_strengths: $shell_inhibition_strengths"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
    echo "planned_cases:"
    echo "1. seed99_shell_dynamics_half_inhibition"
    echo "2. seed42_shell_dynamics_half_inhibition"
    echo "3. seed7_shell_dynamics_half_inhibition"
    echo

    run_case "seed99_shell_dynamics_half_inhibition" 99
    run_case "seed42_shell_dynamics_half_inhibition" 42
    run_case "seed7_shell_dynamics_half_inhibition" 7

    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
