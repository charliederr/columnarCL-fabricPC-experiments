#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
runner="${repo_root}/scripts/run_codex_cifar10_depth_spanning.sh"
seed="${1:-42}"
num_epochs="${2:-8}"
lr="${3:-0.005}"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="${repo_root}/results/codex_shell_composer_diagnostic_comparison_seed${seed}_${hostname_value}_${timestamp}.log"

cd "$repo_root"
mkdir -p results

run_case() {
    local case_name="$1"
    local shell_readout_mode="$2"
    local shell_bridge_mode="$3"

    echo "======================================================================"
    echo "Starting case=$case_name at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "seed: $seed"
    echo "lr: $lr"
    echo "num_epochs: $num_epochs"
    echo "diagnose_mode: composer_shells"
    echo "combiner: shell_attention"
    echo "column_teacher_weight: 0.0"
    echo "shell_teacher_weights: 0,0,0,0"
    echo "column_shell_teacher_weights: 0,0,0,0"
    echo "column_shell_readout: $shell_readout_mode"
    echo "column_shell_bridge: $shell_bridge_mode"
    echo "readout_mode: nobypass"
    echo "======================================================================"

    bash "$runner" "$seed" "$lr" composer_shells "$num_epochs" 0.0 0,0,0,0 nobypass 0,0,0,0 "$shell_readout_mode" "$shell_bridge_mode" shell_attention

    echo "Finished case=$case_name at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
}

{
    echo "repo: $repo_root"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --short --branch
    echo "runner: $runner"
    echo "seed: $seed"
    echo "lr: $lr"
    echo "num_epochs: $num_epochs"
    echo "diagnose_mode: composer_shells"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
    echo "planned_cases:"
    echo "1. shell_composer_with_direct_readout_and_bridge"
    echo "2. shell_composer_composer_only"
    echo

    run_case "shell_composer_with_direct_readout_and_bridge" on on
    run_case "shell_composer_composer_only" off off

    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
