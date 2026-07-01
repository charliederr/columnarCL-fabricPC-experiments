#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
python_bin="${FPC_PYTHON:-/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python}"
seed="${1:-42}"
lr="${2:-0.01}"
diagnose_mode="${3:-nodiag}"
num_epochs="${4:-10}"
column_teacher_weight="${5:-0.1}"
shell_teacher_weights="${6:-0,0,0,0}"
readout_mode="${7:-bypass}"
column_shell_teacher_weights="${8:-0,0,0,0}"
column_shell_readout_mode="${9:-off}"
column_shell_bridge_mode="${10:-off}"
combiner_mode="${11:-sum}"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
lr_label="${lr//./p}"
epochs_label="${num_epochs//./p}"
teacher_weight_label="${column_teacher_weight//./p}"
shell_weights_label="${shell_teacher_weights//./p}"
shell_weights_label="${shell_weights_label//,/_}"
column_shell_weights_label="${column_shell_teacher_weights//./p}"
column_shell_weights_label="${column_shell_weights_label//,/_}"
diagnose_label="nodiag"
extra_args=()
readout_label="bypass"
readout_args=()
column_shell_readout_label="off"
column_shell_readout_args=()
column_shell_bridge_label="off"
column_shell_bridge_args=()
combiner_label="${combiner_mode//_/-}"

if [[ "$diagnose_mode" == "diagnose" || "$diagnose_mode" == "--diagnose_energy" ]]; then
    diagnose_label="diag"
    extra_args+=(--diagnose_energy)
elif [[ "$diagnose_mode" == "shells" || "$diagnose_mode" == "--diagnose_shells" ]]; then
    diagnose_label="shells"
    extra_args+=(--diagnose_shells)
elif [[ "$diagnose_mode" == "diag_shells" ]]; then
    diagnose_label="diag_shells"
    extra_args+=(--diagnose_energy --diagnose_shells)
fi

if [[ "$readout_mode" == "bypass" || "$readout_mode" == "--bypass_columns" ]]; then
    readout_label="bypass"
    readout_args+=(--bypass_columns)
elif [[ "$readout_mode" == "nobypass" || "$readout_mode" == "columns_only" ]]; then
    readout_label="nobypass"
else
    echo "Unknown readout mode: $readout_mode" >&2
    echo "Use 'bypass' or 'nobypass'." >&2
    exit 2
fi

if [[ "$column_shell_readout_mode" == "on" || "$column_shell_readout_mode" == "true" || "$column_shell_readout_mode" == "shellreadout" ]]; then
    column_shell_readout_label="on"
    column_shell_readout_args+=(--column_shell_readout)
elif [[ "$column_shell_readout_mode" == "off" || "$column_shell_readout_mode" == "false" || "$column_shell_readout_mode" == "noshellreadout" ]]; then
    column_shell_readout_label="off"
else
    echo "Unknown column shell readout mode: $column_shell_readout_mode" >&2
    echo "Use 'on' or 'off'." >&2
    exit 2
fi

if [[ "$column_shell_bridge_mode" == "on" || "$column_shell_bridge_mode" == "true" || "$column_shell_bridge_mode" == "shellbridge" ]]; then
    column_shell_bridge_label="on"
    column_shell_bridge_args+=(--column_shell_bridge)
elif [[ "$column_shell_bridge_mode" == "off" || "$column_shell_bridge_mode" == "false" || "$column_shell_bridge_mode" == "noshellbridge" ]]; then
    column_shell_bridge_label="off"
else
    echo "Unknown column shell bridge mode: $column_shell_bridge_mode" >&2
    echo "Use 'on' or 'off'." >&2
    exit 2
fi

if [[ "$combiner_mode" != "sum" && "$combiner_mode" != "attention" && "$combiner_mode" != "shell_attention" ]]; then
    echo "Unknown combiner mode: $combiner_mode" >&2
    echo "Use 'sum', 'attention', or 'shell_attention'." >&2
    exit 2
fi

log_path="${repo_root}/results/codex_resnet18_shelldynamicson_combiner${combiner_label}_column_teacher${teacher_weight_label}_shell${shell_weights_label}_colshell${column_shell_weights_label}_colshellreadout${column_shell_readout_label}_colshellbridge${column_shell_bridge_label}_${readout_label}_norm_fixedln_seed${seed}_lr${lr_label}_ep${epochs_label}_${diagnose_label}_${hostname_value}_${timestamp}.log"

cd "$repo_root"
mkdir -p results

{
    echo "repo: $repo_root"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --short --branch
    echo "python: $python_bin"
    echo "seed: $seed"
    echo "lr: $lr"
    echo "num_epochs: $num_epochs"
    echo "diagnose: $diagnose_label"
    echo "column_teacher_head: enabled"
    echo "column_teacher_weight: $column_teacher_weight"
    echo "shell_teacher_weights: $shell_teacher_weights"
    echo "column_shell_teacher_weights: $column_shell_teacher_weights"
    echo "column_shell_readout: $column_shell_readout_label"
    echo "column_shell_bridge: $column_shell_bridge_label"
    echo "combiner: $combiner_mode"
    echo "shell_evidence_cascade: on"
    echo "shell_evidence_cascade_scale: 0.05,0.05,0.05"
    echo "shell_inhibition_strengths: 0,0.35,0.22,0.10"
    echo "readout_mode: $readout_label"
    "$python_bin" -c "import jax; print(jax.devices()); print(jax.default_backend())"
    "$python_bin" scripts/train_cifar10_depth_spanning.py \
        --model resnet18 \
        --activation leaky_relu \
        --column_activation leaky_relu \
        --num_columns 4 \
        --num_shared 2 \
        --active_nonshared 2 \
        --column_mode all_active \
        --combiner "$combiner_mode" \
        --embed_dim 64 \
        --microcolumn_dim 32 \
        --batch_size 128 \
        --num_epochs "$num_epochs" \
        --lr "$lr" \
        --weight_decay 0.01 \
        --infer_steps 40 \
        --eta_infer 0.1 \
        --infer_max_norm 1.0 \
        --eval_every 1 \
        --seed "$seed" \
        --layer_norm_tokens \
        --fix_ln_gamma \
        --column_teacher_weight "$column_teacher_weight" \
        --shell_teacher_weights "$shell_teacher_weights" \
        --column_shell_teacher_weights "$column_shell_teacher_weights" \
        "${column_shell_readout_args[@]}" \
        "${column_shell_bridge_args[@]}" \
        "${readout_args[@]}" \
        "${extra_args[@]}"
} 2>&1 | tee "$log_path"

echo "log_path: $log_path"
