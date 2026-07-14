#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

seed="${1:-42}"
num_epochs="${2:-10}"
lr="${3:-0.005}"
diagnose_mode="${4:-composer_shells}"

bash "$script_dir/run_codex_cifar10_depth_spanning.sh" \
    "$seed" \
    "$lr" \
    "$diagnose_mode" \
    "$num_epochs" \
    "0.0" \
    "0,0,0,0" \
    "nobypass" \
    "0,0,0,0" \
    "off" \
    "on" \
    "shell_attention" \
    "on" \
    "10" \
    "3" \
    "7" \
    "1,1.5,2,3" \
    "0.0" \
    "off" \
    "0.0" \
    "0.0" \
    "off" \
    "stage3" \
    "96" \
    "32" \
    "spatial_reference" \
    "1.0" \
    "16"
