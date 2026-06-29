#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
runner="${repo_root}/scripts/run_codex_cifar10_depth_spanning.sh"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="${repo_root}/results/codex_shell_bridge_readout_replicate_sweep_${hostname_value}_${timestamp}.log"

cd "$repo_root"
mkdir -p results

run_case() {
    local case_name="$1"
    local seed="$2"

    echo "======================================================================"
    echo "Starting case=$case_name at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "seed: $seed"
    echo "lr: 0.005"
    echo "diagnose_mode: shells"
    echo "num_epochs: 20"
    echo "column_teacher_weight: 0.0"
    echo "shell_teacher_weights: 0,0,0,0"
    echo "readout_mode: nobypass"
    echo "column_shell_teacher_weights: 0,0,0,0"
    echo "column_shell_readout_mode: on"
    echo "column_shell_bridge_mode: on"
    echo "======================================================================"

    bash "$runner" \
        "$seed" \
        0.005 \
        shells \
        20 \
        0.0 \
        0,0,0,0 \
        nobypass \
        0,0,0,0 \
        on \
        on

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
    echo "1. seed42_shell_bridge_plus_direct_readout_zero_teacher_energy"
    echo "2. seed99_shell_bridge_plus_direct_readout_zero_teacher_energy"
    echo "3. seed7_shell_bridge_plus_direct_readout_zero_teacher_energy"
    echo

    run_case "seed42_shell_bridge_plus_direct_readout_zero_teacher_energy" 42
    run_case "seed99_shell_bridge_plus_direct_readout_zero_teacher_energy" 99
    run_case "seed7_shell_bridge_plus_direct_readout_zero_teacher_energy" 7

    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
