
#!/usr/bin/env bash
set -euo pipefail

MAX_RETRIES=500
RETRY_COUNT=0
TIMEOUT_SECONDS=3000
DATASET="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
LOCAL_DIR="/mnt/petrelfs/share/efm_p/wangfangjing/datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim-hf"

while (( RETRY_COUNT < MAX_RETRIES )); do
    echo "Attempt $((RETRY_COUNT+1)) of $MAX_RETRIES..."
    
    timeout $TIMEOUT_SECONDS huggingface-cli download \
      --repo-type dataset \
      "$DATASET" \
      --include "**/*_1000/**" \
      --local-dir "$LOCAL_DIR" 

    DOWNLOAD_EXIT_CODE=$?
    
    # 检查下载结果
    if [ $DOWNLOAD_EXIT_CODE -eq 0 ]; then
        echo "✅ 下载成功完成！"
        echo "完成时间: $(date)"
        exit 0
    elif [ $DOWNLOAD_EXIT_CODE -eq 124 ]; then
        echo "⏰ 下载超时（${TIMEOUT_SECONDS}秒），准备重试..."
    else
        echo "❌ 下载失败，退出代码: $DOWNLOAD_EXIT_CODE"
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "等待 $WAIT_SECONDS 秒后重试..."
        sleep $WAIT_SECONDS
        echo "================================"
    fi
done

echo "❌ Failed after $MAX_RETRIES attempts."
exit 1
