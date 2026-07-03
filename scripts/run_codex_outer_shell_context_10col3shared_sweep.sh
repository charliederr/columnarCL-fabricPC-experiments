#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
runner="${repo_root}/scripts/run_codex_cifar10_depth_spanning.sh"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="${repo_root}/results/codex_outer_shell_context_10col3shared_sweep_${hostname_value}_${timestamp}.log"

cd "$repo_root"
mkdir -p results

run_case() {
    local case_name="$1"
    local seed="$2"

    echo "======================================================================"
    echo "Starting case=$case_name at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "seed: $seed"
    echo "num_columns: 10"
    echo "num_shared: 3"
    echo "active_nonshared: 7"
    echo "lr: 0.005"
    echo "num_epochs: 20"
    echo "diagnose_mode: composer_shells"
    echo "combiner: shell_attention"
    echo "shell_evidence_cascade: on"
    echo "shell_evidence_cascade_scale: 0.05,0.05,0.05"
    echo "shell_inhibition_strengths: 0,0.35,0.22,0.10"
    echo "column_teacher_weight: 0.0"
    echo "shell_teacher_weights: 0,0,0,0"
    echo "column_shell_teacher_weights: 0,0,0,0.002"
    echo "column_shell_readout: off"
    echo "column_shell_bridge: off"
    echo "outer_shell_context: on"
    echo "readout_mode: nobypass"
    echo "======================================================================"

    bash "$runner" \
        "$seed" \
        0.005 \
        composer_shells \
        20 \
        0.0 \
        0,0,0,0 \
        nobypass \
        0,0,0,0.002 \
        off \
        off \
        shell_attention \
        on \
        10 \
        3 \
        7

    echo "Finished case=$case_name at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
}

{
    echo "repo: $repo_root"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --short --branch
    echo "runner: $runner"
    echo "combiner: shell_attention"
    echo "diagnose_mode: composer_shells"
    echo "num_columns: 10"
    echo "num_shared: 3"
    echo "active_nonshared: 7"
    echo "shell_evidence_cascade: on"
    echo "shell_evidence_cascade_scale: 0.05,0.05,0.05"
    echo "shell_inhibition_strengths: 0,0.35,0.22,0.10"
    echo "column_teacher_weight: 0.0"
    echo "shell_teacher_weights: 0,0,0,0"
    echo "column_shell_teacher_weights: 0,0,0,0.002"
    echo "column_shell_readout: off"
    echo "column_shell_bridge: off"
    echo "outer_shell_context: on"
    echo "readout_mode: nobypass"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
    echo "planned_cases:"
    echo "1. seed42_outer_shell_context_10col3shared"
    echo "2. seed99_outer_shell_context_10col3shared"
    echo "3. seed7_outer_shell_context_10col3shared"
    echo

    run_case "seed42_outer_shell_context_10col3shared" 42
    run_case "seed99_outer_shell_context_10col3shared" 99
    run_case "seed7_outer_shell_context_10col3shared" 7

    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "master_log: $master_log"
} 2>&1 | tee "$master_log"
