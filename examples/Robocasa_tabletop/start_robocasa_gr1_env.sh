#!/bin/bash
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



your_ckpt=./results/Checkpoints/1025_qwenGR00T_fourier_gr1_unified_1000_gpus/checkpoints/steps_20000_pytorch_model.pt
sim_python=~/miniconda3/envs/starVLA/bin/python
port=5678
# DEBUG=true


CUDA_VISIBLE_DEVICES=2 ${sim_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16


echo `which python`

export SimplerEnv_PATH=~/Projects/SimplerEnv
export PYTHONPATH=~/Envs/miniconda3/envs/dinoact:${PYTHONPATH}
export PYTHONPATH=$(pwd):${PYTHONPATH}

MODEL_PATH=./results/Checkpoints/1003_qwenoft/checkpoints/steps_100000_pytorch_model.pt
# MODEL_PATH=$1
ckpt_path=${MODEL_PATH}
TSET_NUM=1
# DEBUG=1

port=5678

IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#CUDA_DEVICES[@]} 

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICES: ${CUDA_DEVICES[@]}"
echo "NUM_GPUS: $NUM_GPUS"

CUDA_VISIBLE_DEVICES=2 ${SimplerEnv_PATH} /mnt/petrelfs/wangfangjing/code/starVLA/examples/Robocasa_tabletop/simulation_env.py \
    --port ${port} \
    --env_name ${ENV_NAMES[0]} \
    --n_episodes 50 \
    --n_envs 10 \
    --max_episode_steps 720 \
    --n_action_steps 1 \
    --video_out_path ${ckpt_path}/${ENV_NAMES[0]} \
    --pretrained_path ${your_ckpt} \
