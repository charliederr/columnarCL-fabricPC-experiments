#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ni/repos/fpc/columnarCL-fabricPC-experiments"
hostname_value="$(hostname)"
timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="${repo_root}/results/codex_outer_context_teacher_10col3shared_${hostname_value}_${timestamp}.log"

run_case() {
    local context_teacher_weight="$1"
    local seed="$2"

    echo
    echo "======================================================================"
    echo "Starting outer_context_teacher${context_teacher_weight}_seed${seed} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "outer_shell_context_teacher_weight: ${context_teacher_weight}"
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
        1,1.5,2,3 \
        "$context_teacher_weight"
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
    echo "shell_lr_multipliers: 1,1.5,2,3"
    echo "column_teacher_weight: 0.0"
    echo "shell_teacher_weights: 0,0,0,0"
    echo "column_shell_teacher_weights: 0,0,0,0"
    echo "context_teacher_weights: 0.0005 0.001"
    echo "baseline_context_teacher_weight: 0.0"
    echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

    run_case 0.0005 42
    run_case 0.0005 99
    run_case 0.0005 7

    run_case 0.001 42
    run_case 0.001 99
    run_case 0.001 7

    echo
    echo "completed_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
} 2>&1 | tee "$master_log"

echo "master_log: $master_log"
