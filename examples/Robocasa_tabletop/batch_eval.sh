#!/bin/bash

# 定义评估函数
EvalEnv() {
    local gpu_id=$1
    local port=$2
    local env_name=$3
    local your_ckpt=$4
    local log_dir=$5
    local robocasa_env_path=$6
    
    # 创建视频输出目录
    local video_out_path="${your_ckpt}.log/n_action_steps_12_max_episode_steps_720_${env_name}"
    mkdir -p "${video_out_path}"
    
    echo "  启动评估: GPU ${gpu_id} (端口 ${port}) - 环境 ${env_name}"
    CUDA_VISIBLE_DEVICES=${gpu_id} ${robocasa_env_path} examples/Robocasa_tabletop/simulation_env.py \
        --args.env_name "${env_name}" \
        --args.port "${port}" \
        --args.n_episodes 50 \
        --args.n_envs 1 \
        --args.max_episode_steps 720 \
        --args.n_action_steps 12 \
        --args.video_out_path "${video_out_path}" \
        --args.pretrained_path "${your_ckpt}" \
        > "${log_dir}/eval_env_${env_name//\//_}_gpu${gpu_id}.log" 2>&1
}

# 定义所有需要评估的环境
declare -a ENV_NAMES=(
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

# 配置参数
your_ckpt=/mnt/petrelfs/wangfangjing/code/starVLA/results/Checkpoints/1106_qwenGR00T_fourier_gr1_unified_1000_fromPT50K_gpus/checkpoints/steps_100000_pytorch_model.pt
sim_python=~/miniconda3/envs/starVLA/bin/python
robocasa_env_path=~/miniconda3/envs/robocasa/bin/python
base_port=5678
NUM_GPUS=8

# 创建日志目录
log_dir="${your_ckpt}.log/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${log_dir}

echo "=== 启动多GPU评估 ==="
echo "GPU数量: ${NUM_GPUS}"
echo "环境数量: ${#ENV_NAMES[@]}"
echo "检查点路径: ${your_ckpt}"
echo "日志目录: ${log_dir}"

# 步骤1: 在8个GPU上启动server_policy
echo ""
echo "步骤1: 在 ${NUM_GPUS} 个GPU上启动server_policy..."
declare -a SERVER_PIDS=()

for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    port=$((base_port + gpu_id))
    echo "  在GPU ${gpu_id} (端口 ${port}) 上启动服务器..."
    
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} deployment/model_server/server_policy.py \
        --ckpt_path ${your_ckpt} \
        --port ${port} \
        --use_bf16 \
        > ${log_dir}/server_gpu${gpu_id}_port${port}.log 2>&1 &
    
    SERVER_PIDS[$gpu_id]=$!
    echo "  服务器PID: ${SERVER_PIDS[$gpu_id]}"
done

# 检查所有服务器是否正常运行
echo "检查服务器状态..."
all_running=true
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    if ! kill -0 ${SERVER_PIDS[$gpu_id]} 2>/dev/null; then
        echo "  错误: GPU ${gpu_id} 上的服务器 (PID ${SERVER_PIDS[$gpu_id]}) 未运行!"
        all_running=false
    else
        echo "  GPU ${gpu_id}: 服务器运行中 (PID ${SERVER_PIDS[$gpu_id]})"
    fi
done

if [ "$all_running" = false ]; then
    echo "错误: 部分服务器启动失败，请检查 ${log_dir} 中的日志"
    exit 1
fi

sleep 30  # 等待服务器初始化完成

# 步骤2: 分配环境到GPU并执行评估（保持最多8个并行任务）
echo ""
echo "步骤2: 分配环境到GPU并启动评估..."

count=0  # 任务计数器

for env_name in "${ENV_NAMES[@]}"; do
    # 计算分配的GPU和端口
    gpu_id=$((count % NUM_GPUS))
    port=$((base_port + gpu_id))
    
    # 启动评估任务，每满8个任务等待一次
    if (( (count + 1) % NUM_GPUS == 0 )); then
        EvalEnv ${gpu_id} ${port} "${env_name}" "${your_ckpt}" "${log_dir}" "${robocasa_env_path}"
    else
        EvalEnv ${gpu_id} ${port} "${env_name}" "${your_ckpt}" "${log_dir}" "${robocasa_env_path}" &
    fi
    
    count=$((count + 1))
    
    # 每启动8个任务等待完成
    # if (( count % NUM_GPUS == 0 )); then
    #     echo "  已启动 ${count} 个评估任务，等待当前批次完成..."
    #     wait
    # fi
done

# 等待剩余任务完成
# wait

# # 步骤3: 汇总评估结果
# echo ""
# echo "步骤3: 汇总评估结果..."
# summary_file="${log_dir}/summary.json"
# ${sim_python} gr00t/eval/aggregate_results.py \
#     --log_dir ${log_dir} \
#     --output ${summary_file}

# 步骤4: 关闭所有server_policy进程
echo ""
echo "步骤4: 关闭所有服务器..."
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    pid=${SERVER_PIDS[$gpu_id]}
    if kill -0 ${pid} 2>/dev/null; then
        echo "  停止GPU ${gpu_id} 上的服务器 (PID: ${pid})..."
        kill ${pid}
    fi
done

echo ""
echo "=== 评估完成 ==="
echo "结果保存至: ${your_ckpt}.log/eval_results/"
echo "日志保存至: ${log_dir}"
echo "汇总结果: ${summary_file}"