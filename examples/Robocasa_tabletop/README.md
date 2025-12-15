# 🚀 Robocasa-GR1-Tabletop-Tasks Evaluation

This document provides instructions for reproducing our **experimental results** with [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks).  
The evaluation process consists of two main parts:  

1. Setting up the `robocasa` environment and dependencies.  
2. Running the evaluation by launching services in both `internvla_m1` and `LIBERO` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

---

## 📊 Experimental Results

| Environment                                                                 | GR00T-N1.5 | QwenPI |
|-----------------------------------------------------------------------------|--------------|-------------|
| gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env             | 0.38         | —           |
| gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env       | 0.32         | —           |
| gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env         | 0.60         | —           |
| gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env         | 0.54         | —           |
| gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env           | 0.38         | —           |
| gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env             | 0.50         | —           |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.38 | — |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 0.46 | — |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 0.58 | — |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 0.62 | — |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.28 | — |
| gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.30 | — |
| gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 0.60 | — |
| gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 0.56 | — |
| gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 0.36 | — |
| gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 0.58 | — |
| gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 0.44 | — |
| gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 0.60 | — |
| gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 0.64 | — |
| gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 0.52 | — |
| gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 0.48 | — |
| gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 0.60 | — |
| gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 0.52 | — |
| gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 0.32 | — |
| **Average** | **0.48** | **—** |


## ⬇️ 0. Download Checkpoints
First, download the checkpoints from 
- [LIBERO-Object](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Object)
- [LIBERO-Spatial](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Spatial)
- [LIBERO-Goal](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Goal)
- [LIBERO-Long](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Long)



## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the base `LIBERO` environment.  

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **internvla_m1 environment**: runs the inference server.  
- **LIBERO environment**: runs the simulation.  

### Step 1. Start the server (internvla_m1 environment)

In the first terminal, activate the `internvla_m1` conda environment and run:  

```bash
bash examples/LIBERO/run_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/LIBERO/run_server.sh`  


---

### Step 2. Start the simulation (LIBERO environment)

In the second terminal, activate the `LIBERO` conda environment and run:  

```bash
bash examples/LIBERO/eval_libero.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/LIBERO/eval_libero.sh` to load action unnormalization stats. 



---


# 🚀 LIBERO Training
## 📦 Step0: Download the training dataset
Download the datasets to the playground/Datasets/LEROBOT_LIBERO_DATA directory:
- [LIBERO-spatial] https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
- [LIBERO-object] https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot
- [LIBERO-goal] https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot
- [LIBERO-10] https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot

## 🚀 Step1: Start Training

```bash
bash scripts/run_scripts/run_libero_train.sh
```
