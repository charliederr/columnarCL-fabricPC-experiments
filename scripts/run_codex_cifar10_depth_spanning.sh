#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
python_bin="${FPC_PYTHON:-/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python}"
seed="${1:-42}"
lr="${2:-0.01}"
diagnose_mode="${3:-nodiag}"
num_epochs="${4:-10}"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
lr_label="${lr//./p}"
epochs_label="${num_epochs//./p}"
diagnose_label="nodiag"
extra_args=()

if [[ "$diagnose_mode" == "diagnose" || "$diagnose_mode" == "--diagnose_energy" ]]; then
    diagnose_label="diag"
    extra_args+=(--diagnose_energy)
fi

log_path="${repo_root}/results/codex_resnet18_bypass_norm_fixedln_seed${seed}_lr${lr_label}_ep${epochs_label}_${diagnose_label}_${hostname_value}_${timestamp}.log"

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
        --num_epochs "$num_epochs" \
        --lr "$lr" \
        --weight_decay 0.01 \
        --infer_steps 40 \
        --eta_infer 0.1 \
        --infer_max_norm 1.0 \
        --eval_every 1 \
        --seed "$seed" \
        --bypass_columns \
        --layer_norm_tokens \
        --fix_ln_gamma \
        "${extra_args[@]}"
} 2>&1 | tee "$log_path"

echo "log_path: $log_path"
