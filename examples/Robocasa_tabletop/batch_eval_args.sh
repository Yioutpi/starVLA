#!/bin/bash

# ======================== 参数解析 ========================
your_ckpt_default="/mnt/petrelfs/wangfangjing/code/starVLA/results/Checkpoints/1120_qwenGR00T_fourier_gr1_10K_gpus_wState_pretrain_freezeQwen/checkpoints/steps_300000_pytorch_model.pt"
n_envs_default=1
max_episode_steps_default=720
n_action_steps_default=12

# 解析命令行参数
your_ckpt=${1:-$your_ckpt_default}
n_envs=${2:-$n_envs_default}
max_episode_steps=${3:-$max_episode_steps_default}
n_action_steps=${4:-$n_action_steps_default}

echo "=== 参数配置 ==="
echo "Checkpoint Path: ${your_ckpt}"
echo "n_envs: ${n_envs}"
echo "max_episode_steps: ${max_episode_steps}"
echo "n_action_steps: ${n_action_steps}"
echo "================="

# ======================== 主程序 ========================

# 定义评估函数
EvalEnv() {
    local gpu_id=$1
    local port=$2
    local env_name=$3
    local your_ckpt=$4
    local log_dir=$5
    local robocasa_env_path=$6
    local n_envs=$7
    local max_episode_steps=$8
    local n_action_steps=$9
    
    local video_out_path="${your_ckpt}.log/n_action_steps_${n_action_steps}_max_episode_steps_${max_episode_steps}_n_envs_${n_envs}_${env_name}"
    mkdir -p "${video_out_path}"
    
    echo "  启动评估: GPU ${gpu_id} (端口 ${port}) - 环境 ${env_name}"
    CUDA_VISIBLE_DEVICES=${gpu_id} ${robocasa_env_path} examples/Robocasa_tabletop/simulation_env.py \
        --args.env_name "${env_name}" \
        --args.port "${port}" \
        --args.n_episodes 50 \
        --args.n_envs "${n_envs}" \
        --args.max_episode_steps "${max_episode_steps}" \
        --args.n_action_steps "${n_action_steps}" \
        --args.video_out_path "${video_out_path}" \
        --args.pretrained_path "${your_ckpt}" \
        > "${log_dir}/eval_env_${env_name//\//_}_gpu${gpu_id}.log" 2>&1
}

# 定义环境列表
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

sim_python=~/miniconda3/envs/starVLA/bin/python
robocasa_env_path=~/miniconda3/envs/robocasa/bin/python
base_port=5678
NUM_GPUS=8

log_dir="${your_ckpt}.log/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${log_dir}

echo "=== 启动多GPU评估 ==="
echo "GPU数量: ${NUM_GPUS}"
echo "环境数量: ${#ENV_NAMES[@]}"
echo "日志目录: ${log_dir}"

# 步骤1: 启动 server_policy
declare -a SERVER_PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    port=$((base_port + gpu_id))
    echo "  启动服务器 GPU ${gpu_id} (端口 ${port}) ..."
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} deployment/model_server/server_policy.py \
        --ckpt_path ${your_ckpt} \
        --port ${port} \
        --use_bf16 \
        > ${log_dir}/server_gpu${gpu_id}_port${port}.log 2>&1 &
    SERVER_PIDS[$gpu_id]=$!
done

sleep 30

# 步骤2: 分配环境到GPU
count=0
for env_name in "${ENV_NAMES[@]}"; do
    gpu_id=$((count % NUM_GPUS))
    port=$((base_port + gpu_id))
    if (( (count + 1) % NUM_GPUS == 0 )); then
        EvalEnv ${gpu_id} ${port} "${env_name}" "${your_ckpt}" "${log_dir}" "${robocasa_env_path}" "${n_envs}" "${max_episode_steps}" "${n_action_steps}"
    else
        EvalEnv ${gpu_id} ${port} "${env_name}" "${your_ckpt}" "${log_dir}" "${robocasa_env_path}" "${n_envs}" "${max_episode_steps}" "${n_action_steps}" &
    fi
    count=$((count + 1))
done

# 步骤3: 清理
echo ""
echo "关闭所有服务器..."
for pid in "${SERVER_PIDS[@]}"; do
    kill ${pid} 2>/dev/null && echo "  已关闭 PID ${pid}"
done

echo "=== 完成 ==="
