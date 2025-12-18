

your_ckpt=/mnt/petrelfs/wangfangjing/code/starVLA/results/Checkpoints/1029_qwenGR00T_fourier_gr1_unified_1000_PnPMilkToMicrowaveClose_gpus_woPretrain/checkpoints/steps_20000_pytorch_model.pt
sim_python=~/miniconda3/envs/starVLA/bin/python
port=5679
# DEBUG=true


CUDA_VISIBLE_DEVICES=2 ${sim_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16