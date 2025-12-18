#!/usr/bin/env bash
set -euo pipefail

# 默认路径
DEFAULT_ROOT="/mnt/petrelfs/wangfangjing/code/starVLA/results/Checkpoints/1029_qwenGR00T_fourier_gr1_unified_1000_gpus_wState/checkpoints/steps_40000_pytorch_model.pt.log/n_action_steps_10_gr1_unified"
ROOT="${1:-$DEFAULT_ROOT}"

if [ ! -d "$ROOT" ]; then
    echo "❌ Error: directory '$ROOT' not found."
    exit 1
fi

TMP_FILE=$(mktemp)

printf "Task\tSuccess\tTotal\tSuccessRate\n" > "$TMP_FILE"

total_success=0
total_count=0

for task_dir in "$ROOT"/*_Env; do
    # 如果不是目录则跳过
    [ -d "$task_dir" ] || continue

    task_name=$(basename "$task_dir")
    success_count=$(find "$task_dir" -type f -name '*_success1.mp4' | wc -l | tr -d ' ')
    total=$(find "$task_dir" -type f -name '*.mp4' | wc -l | tr -d ' ')

    if [ "$total" -gt 0 ]; then
        rate=$(awk "BEGIN {printf \"%.2f\", $success_count/$total*100}")
    else
        rate="0.00"
    fi

    printf "%s\t%s\t%s\t%s%%\n" "$task_name" "$success_count" "$total" "$rate" >> "$TMP_FILE"

    total_success=$((total_success + success_count))
    total_count=$((total_count + total))
done

if [ "$total_count" -gt 0 ]; then
    total_rate=$(awk "BEGIN {printf \"%.2f\", $total_success/$total_count*100}")
else
    total_rate="0.00"
fi

{
    echo "---------------------------------------------------------------"
    printf "%s\t%s\t%s\t%s%%\n" "Total" "$total_success" "$total_count" "$total_rate"
} >> "$TMP_FILE"

column -t -s $'\t' "$TMP_FILE"
rm -f "$TMP_FILE"
