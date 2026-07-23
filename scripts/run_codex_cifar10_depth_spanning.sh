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
outer_shell_context_mode="${12:-off}"
num_columns="${13:-4}"
num_shared="${14:-2}"
active_nonshared="${15:-2}"
shell_lr_multipliers="${16:-1,1,1,1}"
outer_shell_context_teacher_weight="${17:-0.0}"
outer_shell_context_evidence_mode="${18:-off}"
outer_shell_context_evidence_teacher_weight="${19:-0.0}"
outer_shell_context_shell_prediction_weight="${20:-0.0}"
outer_shell_context_to_bridge_mode="${21:-off}"
column_grid="${22:-stage4}"
embed_dim="${23:-64}"
microcolumn_dim="${24:-32}"
column_gaussian_energy_mode="${25:-sum}"
column_gaussian_precision="${26:-1.0}"
column_gaussian_reference_sites="${27:-16}"
outer_shell_context_bridge_scale="${28:-0.0}"
column_mode="${29:-all_active}"
support_mask="${30:-}"
post_training_diagnostics="${31:-full}"
inward_shell_promotion_weight="${32:-0.0}"
inward_shell_promotion_pairs="${33:-all}"
inward_shell_promotion_target_gradient_scale="${34:-1.0}"
inward_shell_promotion_warmup_epochs="${35:-0.0}"
inward_shell_promotion_ramp_epochs="${36:-0.0}"
promoted_shell_bridge_mode="${37:-off}"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
lr_label="${lr//./p}"
epochs_label="${num_epochs//./p}"
teacher_weight_label="${column_teacher_weight//./p}"
shell_weights_label="${shell_teacher_weights//./p}"
shell_weights_label="${shell_weights_label//,/_}"
column_shell_weights_label="${column_shell_teacher_weights//./p}"
column_shell_weights_label="${column_shell_weights_label//,/_}"
shell_lr_label="${shell_lr_multipliers//./p}"
shell_lr_label="${shell_lr_label//,/_}"
outer_context_teacher_label="${outer_shell_context_teacher_weight//./p}"
outer_context_evidence_teacher_label="${outer_shell_context_evidence_teacher_weight//./p}"
outer_context_shell_prediction_label="${outer_shell_context_shell_prediction_weight//./p}"
outer_context_bridge_scale_label="${outer_shell_context_bridge_scale//./p}"
inward_shell_promotion_label="${inward_shell_promotion_weight//./p}"
inward_shell_promotion_target_gradient_label="${inward_shell_promotion_target_gradient_scale//./p}"
inward_shell_promotion_warmup_label="${inward_shell_promotion_warmup_epochs//./p}"
inward_shell_promotion_ramp_label="${inward_shell_promotion_ramp_epochs//./p}"
diagnose_label="nodiag"
extra_args=()
readout_label="bypass"
readout_args=()
column_shell_readout_label="off"
column_shell_readout_args=()
column_shell_bridge_label="off"
column_shell_bridge_args=()
promoted_shell_bridge_label="off"
promoted_shell_bridge_args=()
outer_shell_context_label="off"
outer_shell_context_args=()
outer_shell_context_to_bridge_label="off"
outer_shell_context_to_bridge_args=()
outer_shell_context_evidence_label="off"
outer_shell_context_evidence_args=()
column_gaussian_energy_args=()
support_mask_args=()

case "$post_training_diagnostics" in
    core|full)
        ;;
    *)
        echo "Unknown post-training diagnostics mode: $post_training_diagnostics" >&2
        echo "Use 'core' or 'full'." >&2
        exit 2
        ;;
esac

case "$column_mode" in
    all_active)
        column_mode_label="all"
        ;;
    first_sparse)
        column_mode_label="first"
        ;;
    random_sparse)
        column_mode_label="random"
        ;;
    *)
        echo "Unknown column mode: $column_mode" >&2
        echo "Use 'all_active', 'first_sparse', or 'random_sparse'." >&2
        exit 2
        ;;
esac

if [[ -n "$support_mask" && "$support_mask" != "none" ]]; then
    support_mask_args+=(--support_mask "$support_mask")
    support_mask_label="${support_mask//,/}"
else
    support_mask_label="none"
fi

if [[ "$diagnose_mode" == "diagnose" || "$diagnose_mode" == "--diagnose_energy" ]]; then
    diagnose_label="diag"
    extra_args+=(--diagnose_energy)
elif [[ "$diagnose_mode" == "shells" || "$diagnose_mode" == "--diagnose_shells" ]]; then
    diagnose_label="shells"
    extra_args+=(--diagnose_shells)
elif [[ "$diagnose_mode" == "diag_shells" ]]; then
    diagnose_label="diag_shells"
    extra_args+=(--diagnose_energy --diagnose_shells)
elif [[ "$diagnose_mode" == "composer" || "$diagnose_mode" == "--diagnose_composer" ]]; then
    diagnose_label="composer"
    extra_args+=(--diagnose_composer)
elif [[ "$diagnose_mode" == "composer_shells" ]]; then
    diagnose_label="composer_shells"
    extra_args+=(--diagnose_composer --diagnose_shells)
elif [[ "$diagnose_mode" == "diag_shells_composer" ]]; then
    diagnose_label="diag_shells_composer"
    extra_args+=(--diagnose_energy --diagnose_shells --diagnose_composer)
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

if [[ "$outer_shell_context_mode" == "on" || "$outer_shell_context_mode" == "true" || "$outer_shell_context_mode" == "outercontext" ]]; then
    outer_shell_context_label="on"
    outer_shell_context_args+=(--outer_shell_context)
elif [[ "$outer_shell_context_mode" == "off" || "$outer_shell_context_mode" == "false" || "$outer_shell_context_mode" == "nooutercontext" ]]; then
    outer_shell_context_label="off"
else
    echo "Unknown outer shell context mode: $outer_shell_context_mode" >&2
    echo "Use 'on' or 'off'." >&2
    exit 2
fi

if [[ "$outer_shell_context_evidence_mode" == "on" || "$outer_shell_context_evidence_mode" == "true" || "$outer_shell_context_evidence_mode" == "outerevidence" ]]; then
    outer_shell_context_evidence_label="on"
    outer_shell_context_evidence_args+=(--outer_shell_context_evidence)
elif [[ "$outer_shell_context_evidence_mode" == "off" || "$outer_shell_context_evidence_mode" == "false" || "$outer_shell_context_evidence_mode" == "noouterevidence" ]]; then
    outer_shell_context_evidence_label="off"
else
    echo "Unknown outer shell context evidence mode: $outer_shell_context_evidence_mode" >&2
    echo "Use 'on' or 'off'." >&2
    exit 2
fi

if [[ "$outer_shell_context_to_bridge_mode" == "on" || "$outer_shell_context_to_bridge_mode" == "true" || "$outer_shell_context_to_bridge_mode" == "outerbridge" ]]; then
    outer_shell_context_to_bridge_label="on"
    outer_shell_context_to_bridge_args+=(--outer_shell_context_to_bridge)
elif [[ "$outer_shell_context_to_bridge_mode" == "off" || "$outer_shell_context_to_bridge_mode" == "false" || "$outer_shell_context_to_bridge_mode" == "noouterbridge" ]]; then
    outer_shell_context_to_bridge_label="off"
else
    echo "Unknown outer shell context to bridge mode: $outer_shell_context_to_bridge_mode" >&2
    echo "Use 'on' or 'off'." >&2
    exit 2
fi

if [[ "$combiner_mode" != "sum" && "$combiner_mode" != "attention" && "$combiner_mode" != "shell_attention" ]]; then
    echo "Unknown combiner mode: $combiner_mode" >&2
    echo "Use 'sum', 'attention', or 'shell_attention'." >&2
    exit 2
fi

case "$combiner_mode" in
    sum)
        combiner_label="sum"
        ;;
    attention)
        combiner_label="att"
        ;;
    shell_attention)
        combiner_label="shatt"
        ;;
esac

if [[ "$column_grid" != "stage4" && "$column_grid" != "stage3" && "$column_grid" != "stage2" ]]; then
    echo "Unknown column grid: $column_grid" >&2
    echo "Use 'stage4', 'stage3', or 'stage2'." >&2
    exit 2
fi

column_grid_label="${column_grid}"

if [[ "$column_gaussian_energy_mode" == "sum" ]]; then
    column_gaussian_energy_label="sum"
    column_gaussian_energy_short="gsum"
elif [[ "$column_gaussian_energy_mode" == "mean" ]]; then
    column_gaussian_energy_label="mean"
    column_gaussian_energy_short="gmean"
elif [[ "$column_gaussian_energy_mode" == "spatial_reference" ]]; then
    column_gaussian_energy_label="spatial_reference"
    column_gaussian_energy_short="gspref"
else
    echo "Unknown column Gaussian energy mode: $column_gaussian_energy_mode" >&2
    echo "Use 'sum', 'mean', or 'spatial_reference'." >&2
    exit 2
fi
column_gaussian_energy_args+=(--column_gaussian_energy_mode "$column_gaussian_energy_mode")
column_gaussian_precision_label="${column_gaussian_precision//./p}"
column_gaussian_reference_sites_label="${column_gaussian_reference_sites//./p}"

if [[ "$readout_label" == "bypass" ]]; then
    readout_short="byp"
else
    readout_short="nobyp"
fi

if [[ "$column_shell_readout_label" == "on" ]]; then
    column_shell_readout_short="sr1"
else
    column_shell_readout_short="sr0"
fi

if [[ "$column_shell_bridge_label" == "on" ]]; then
    column_shell_bridge_short="br1"
else
    column_shell_bridge_short="br0"
fi

if [[ "$promoted_shell_bridge_mode" == "on" || "$promoted_shell_bridge_mode" == "true" || "$promoted_shell_bridge_mode" == "promotedbridge" ]]; then
    promoted_shell_bridge_label="on"
    promoted_shell_bridge_args+=(--promoted_shell_bridge)
elif [[ "$promoted_shell_bridge_mode" == "off" || "$promoted_shell_bridge_mode" == "false" || "$promoted_shell_bridge_mode" == "nopromotedbridge" ]]; then
    promoted_shell_bridge_label="off"
else
    echo "Unknown promoted shell bridge mode: $promoted_shell_bridge_mode" >&2
    echo "Use 'on' or 'off'." >&2
    exit 2
fi

if [[ "$promoted_shell_bridge_label" == "on" ]]; then
    promoted_shell_bridge_short="pbr1"
else
    promoted_shell_bridge_short="pbr0"
fi

if [[ "$outer_shell_context_label" == "on" ]]; then
    outer_shell_context_short="oc1"
else
    outer_shell_context_short="oc0"
fi

if [[ "$outer_shell_context_evidence_label" == "on" ]]; then
    outer_shell_context_evidence_short="oce1"
else
    outer_shell_context_evidence_short="oce0"
fi

if [[ "$outer_shell_context_to_bridge_label" == "on" ]]; then
    outer_shell_context_to_bridge_short="ocb1"
else
    outer_shell_context_to_bridge_short="ocb0"
fi

# Capacity and context labels made the original descriptive filename exceed
# common 255-byte filename limits. The full configuration is still written in
# the log header; the filename keeps only compact run identifiers.
if [[ "$inward_shell_promotion_warmup_epochs" != "0.0" || "$inward_shell_promotion_ramp_epochs" != "0.0" ]]; then
    log_path="${repo_root}/results/codex_dspan_sched_${num_columns}c_${num_shared}s_${active_nonshared}a_${combiner_label}_isp${inward_shell_promotion_label}_iptg${inward_shell_promotion_target_gradient_label}_ipw${inward_shell_promotion_warmup_label}_ipr${inward_shell_promotion_ramp_label}_${promoted_shell_bridge_short}_seed${seed}_lr${lr_label}_ep${epochs_label}_${diagnose_label}_${hostname_value}_${timestamp}.log"
else
    log_path="${repo_root}/results/codex_dspan_${num_columns}c_${num_shared}s_${active_nonshared}a_${combiner_label}_${column_grid_label}_isp${inward_shell_promotion_label}_iptg${inward_shell_promotion_target_gradient_label}_${column_shell_bridge_short}_${promoted_shell_bridge_short}_seed${seed}_lr${lr_label}_ep${epochs_label}_${diagnose_label}_${hostname_value}_${timestamp}.log"
fi

cd "$repo_root"
mkdir -p results

{
    echo "repo: $repo_root"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --short --branch
    echo "python: $python_bin"
    echo "seed: $seed"
    echo "num_columns: $num_columns"
    echo "num_shared: $num_shared"
    echo "active_nonshared: $active_nonshared"
    echo "column_mode: $column_mode"
    echo "support_mask: ${support_mask:-none}"
    echo "lr: $lr"
    echo "column_grid: $column_grid"
    echo "embed_dim: $embed_dim"
    echo "microcolumn_dim: $microcolumn_dim"
    echo "column_gaussian_energy_mode: $column_gaussian_energy_label"
    echo "column_gaussian_precision: $column_gaussian_precision"
    echo "column_gaussian_reference_sites: $column_gaussian_reference_sites"
    echo "shell_lr_multipliers: $shell_lr_multipliers"
    echo "num_epochs: $num_epochs"
    echo "diagnose: $diagnose_label"
    echo "post_training_diagnostics: $post_training_diagnostics"
    echo "column_teacher_head: enabled"
    echo "column_teacher_weight: $column_teacher_weight"
    echo "shell_teacher_weights: $shell_teacher_weights"
    echo "column_shell_teacher_weights: $column_shell_teacher_weights"
    echo "column_shell_readout: $column_shell_readout_label"
    echo "column_shell_bridge: $column_shell_bridge_label"
    echo "promoted_shell_bridge: $promoted_shell_bridge_label"
    echo "outer_shell_context: $outer_shell_context_label"
    echo "outer_shell_context_to_bridge: $outer_shell_context_to_bridge_label"
    echo "outer_shell_context_bridge_scale: $outer_shell_context_bridge_scale"
    echo "outer_shell_context_teacher_weight: $outer_shell_context_teacher_weight"
    echo "outer_shell_context_shell_prediction_weight: $outer_shell_context_shell_prediction_weight"
    echo "inward_shell_promotion_weight: $inward_shell_promotion_weight"
    echo "inward_shell_promotion_pairs: $inward_shell_promotion_pairs"
    echo "inward_shell_promotion_target_gradient_scale: $inward_shell_promotion_target_gradient_scale"
    echo "inward_shell_promotion_warmup_epochs: $inward_shell_promotion_warmup_epochs"
    echo "inward_shell_promotion_ramp_epochs: $inward_shell_promotion_ramp_epochs"
    echo "outer_shell_context_evidence: $outer_shell_context_evidence_label"
    echo "outer_shell_context_evidence_teacher_weight: $outer_shell_context_evidence_teacher_weight"
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
        --num_columns "$num_columns" \
        --num_shared "$num_shared" \
        --active_nonshared "$active_nonshared" \
        --column_mode "$column_mode" \
        "${support_mask_args[@]}" \
        --combiner "$combiner_mode" \
        --column_grid "$column_grid" \
        --embed_dim "$embed_dim" \
        --microcolumn_dim "$microcolumn_dim" \
        --batch_size 128 \
        --num_epochs "$num_epochs" \
        --lr "$lr" \
        --weight_decay 0.01 \
        --shell_lr_multipliers "$shell_lr_multipliers" \
        --infer_steps 40 \
        --eta_infer 0.1 \
        --infer_max_norm 1.0 \
        "${column_gaussian_energy_args[@]}" \
        --column_gaussian_precision "$column_gaussian_precision" \
        --column_gaussian_reference_sites "$column_gaussian_reference_sites" \
        --eval_every 1 \
        --seed "$seed" \
        --post_training_diagnostics "$post_training_diagnostics" \
        --layer_norm_tokens \
        --fix_ln_gamma \
        --column_teacher_weight "$column_teacher_weight" \
        --shell_teacher_weights "$shell_teacher_weights" \
        --column_shell_teacher_weights "$column_shell_teacher_weights" \
        --outer_shell_context_teacher_weight "$outer_shell_context_teacher_weight" \
        --outer_shell_context_shell_prediction_weight "$outer_shell_context_shell_prediction_weight" \
        --inward_shell_promotion_weight "$inward_shell_promotion_weight" \
        --inward_shell_promotion_pairs "$inward_shell_promotion_pairs" \
        --inward_shell_promotion_target_gradient_scale "$inward_shell_promotion_target_gradient_scale" \
        --inward_shell_promotion_warmup_epochs "$inward_shell_promotion_warmup_epochs" \
        --inward_shell_promotion_ramp_epochs "$inward_shell_promotion_ramp_epochs" \
        --outer_shell_context_bridge_scale "$outer_shell_context_bridge_scale" \
        --outer_shell_context_evidence_teacher_weight "$outer_shell_context_evidence_teacher_weight" \
        "${column_shell_readout_args[@]}" \
        "${column_shell_bridge_args[@]}" \
        "${promoted_shell_bridge_args[@]}" \
        "${outer_shell_context_args[@]}" \
        "${outer_shell_context_to_bridge_args[@]}" \
        "${outer_shell_context_evidence_args[@]}" \
        "${readout_args[@]}" \
        "${extra_args[@]}"
} 2>&1 | tee "$log_path"

echo "log_path: $log_path"
