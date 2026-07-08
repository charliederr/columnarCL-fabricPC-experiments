#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="${repo_root}/results/codex_shell_lr_profile_outer_context_10col3shared_${hostname_value}_${timestamp}.log"

run_case() {
    local profile_name="$1"
    local shell_lr_multipliers="$2"
    local seed="$3"

    echo
    echo "======================================================================"
    echo "Starting ${profile_name}_seed${seed} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "shell_lr_multipliers: ${shell_lr_multipliers}"
    echo "======================================================================"

    "${repo_root}/scripts/run_codex_cifar10_depth_spanning.sh" \
        "$seed" \
        0.005 \
        composer_shells \
        20 \
        0.0 \
        0,0,0,0 \
        nobypass \
        0,0,0,0 \
        off \
        off \
        shell_attention \
        on \
        10 \
        3 \
        7 \
        "$shell_lr_multipliers"
}

cd "$repo_root"
mkdir -p results

{
    echo "repo: $repo_root"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --short --branch
    echo "seeds: 42 99 7"
    echo "lr: 0.005"
    echo "num_epochs: 20"
    echo "num_columns: 10"
    echo "num_shared: 3"
    echo "active_nonshared: 7"
    echo "combiner: shell_attention"
    echo "outer_shell_context: on"
    echo "column_teacher_weight: 0.0"
    echo "shell_teacher_weights: 0,0,0,0"
    echo "column_shell_teacher_weights: 0,0,0,0"
    echo "baseline_shell_lr_multipliers: 1,1.5,2,3"
    echo "flat_control_shell_lr_multipliers: 1,1,1,1"
    echo "profile_gentler_shell_lr_multipliers: 1,1.25,1.75,2.5"
    echo "profile_stronger_shell_lr_multipliers: 1,2,3,4"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

    run_case "gentler_shell_lr_outer_context_no_teachers" 1,1.25,1.75,2.5 42
    run_case "gentler_shell_lr_outer_context_no_teachers" 1,1.25,1.75,2.5 99
    run_case "gentler_shell_lr_outer_context_no_teachers" 1,1.25,1.75,2.5 7

    run_case "stronger_shell_lr_outer_context_no_teachers" 1,2,3,4 42
    run_case "stronger_shell_lr_outer_context_no_teachers" 1,2,3,4 99
    run_case "stronger_shell_lr_outer_context_no_teachers" 1,2,3,4 7

    echo
    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
} 2>&1 | tee "$master_log"

echo "master_log: $master_log"
