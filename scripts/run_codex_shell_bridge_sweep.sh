#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
runner="${repo_root}/scripts/run_codex_cifar10_depth_spanning.sh"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="${repo_root}/results/codex_shell_bridge_sweep_${hostname_value}_${timestamp}.log"

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
    local column_shell_bridge_mode="${11}"

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
    echo "column_shell_bridge_mode: $column_shell_bridge_mode"
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
        "$column_shell_readout_mode" \
        "$column_shell_bridge_mode"

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
    echo "1. seed42_shell_bridge_only_no_teachers"
    echo "2. seed42_shell_bridge_plus_direct_readout_no_teachers"
    echo "3. seed99_shell_bridge_only_no_teachers"
    echo "4. seed7_shell_bridge_only_no_teachers"
    echo

    run_case "seed42_shell_bridge_only_no_teachers" \
        42 \
        0.005 \
        shells \
        20 \
        0.0 \
        0,0,0,0 \
        nobypass \
        0,0,0,0 \
        off \
        on

    run_case "seed42_shell_bridge_plus_direct_readout_no_teachers" \
        42 \
        0.005 \
        shells \
        20 \
        0.0 \
        0,0,0,0 \
        nobypass \
        0,0,0,0 \
        on \
        on

    run_case "seed99_shell_bridge_only_no_teachers" \
        99 \
        0.005 \
        shells \
        20 \
        0.0 \
        0,0,0,0 \
        nobypass \
        0,0,0,0 \
        off \
        on

    run_case "seed7_shell_bridge_only_no_teachers" \
        7 \
        0.005 \
        shells \
        20 \
        0.0 \
        0,0,0,0 \
        nobypass \
        0,0,0,0 \
        off \
        on

    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
