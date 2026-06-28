#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
runner="${repo_root}/scripts/run_codex_cifar10_depth_spanning.sh"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="${repo_root}/results/codex_shell_readout_overnight_sweep_${hostname_value}_${timestamp}.log"

cd "$repo_root"
mkdir -p results

run_case() {
    local case_name="$1"
    local seed="$2"
    local lr="$3"
    local diagnose_mode="$4"
    local num_epochs="$5"
    local column_teacher_weight="$6"
    local shell_teacher_weights="$7"
    local readout_mode="$8"
    local column_shell_teacher_weights="$9"
    local column_shell_readout_mode="${10}"

    echo "======================================================================"
    echo "Starting case=$case_name at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "seed: $seed"
    echo "lr: $lr"
    echo "diagnose_mode: $diagnose_mode"
    echo "num_epochs: $num_epochs"
    echo "column_teacher_weight: $column_teacher_weight"
    echo "shell_teacher_weights: $shell_teacher_weights"
    echo "readout_mode: $readout_mode"
    echo "column_shell_teacher_weights: $column_shell_teacher_weights"
    echo "column_shell_readout_mode: $column_shell_readout_mode"
    echo "======================================================================"

    bash "$runner" \
        "$seed" \
        "$lr" \
        "$diagnose_mode" \
        "$num_epochs" \
        "$column_teacher_weight" \
        "$shell_teacher_weights" \
        "$readout_mode" \
        "$column_shell_teacher_weights" \
        "$column_shell_readout_mode"

    echo "Finished case=$case_name at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
}

{
    echo "repo: $repo_root"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --short --branch
    echo "runner: $runner"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
    echo "planned_cases:"
    echo "1. seed42_no_column_shell_teachers"
    echo "2. seed42_outer_shell_teacher_heavier"
    echo "3. seed99_current_shell_readout_replicate"
    echo "4. seed7_current_shell_readout_replicate"
    echo

    run_case "seed42_no_column_shell_teachers" \
        42 \
        0.005 \
        shells \
        20 \
        0.0 \
        0,0,0,0 \
        nobypass \
        0,0,0,0 \
        on

    run_case "seed42_outer_shell_teacher_heavier" \
        42 \
        0.005 \
        shells \
        20 \
        0.0 \
        0,0,0,0 \
        nobypass \
        0.001,0.001,0.002,0.006 \
        on

    run_case "seed99_current_shell_readout_replicate" \
        99 \
        0.005 \
        shells \
        20 \
        0.0 \
        0,0,0,0 \
        nobypass \
        0.001,0.001,0.002,0.002 \
        on

    run_case "seed7_current_shell_readout_replicate" \
        7 \
        0.005 \
        shells \
        20 \
        0.0 \
        0,0,0,0 \
        nobypass \
        0.001,0.001,0.002,0.002 \
        on

    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
