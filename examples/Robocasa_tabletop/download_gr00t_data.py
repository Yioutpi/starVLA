from huggingface_hub import list_repo_files, hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import random

repo = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
local_dir = "/mnt/petrelfs/share/efm_p/wangfangjing/datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"

folders = [
    "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
]

print("Listing all files in repository...")
all_files = list_repo_files(repo_id=repo, repo_type="dataset")
target_files = [f for f in all_files if any(f.startswith(folder + "/") for folder in folders)]
print(f"Found {len(target_files)} matching files to download.\n")

# 单文件下载函数，带重试
def download_with_retry(filename, max_retries=50):
    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            return True
        except Exception as e:
            wait_time = random.uniform(1, 5) * attempt / 10  # 随机退避，防止限流
            print(f"⚠️ [{attempt}/{max_retries}] Failed: {filename} ({e}); retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
    print(f"❌ Giving up after {max_retries} retries: {filename}")
    return False

# 并行下载
max_workers = 32
print(f"Starting parallel download with {max_workers} workers...\n")

failed_files = []

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(download_with_retry, f): f for f in target_files}
    for i, future in enumerate(as_completed(futures), 1):
        fname = futures[future]
        try:
            ok = future.result()
            if ok:
                print(f"✅ [{i}/{len(target_files)}] {fname}")
            else:
                failed_files.append(fname)
        except Exception as e:
            print(f"❌ Unexpected error for {fname}: {e}")
            failed_files.append(fname)

print("\nAll downloads attempted.")
if failed_files:
    print(f"❌ {len(failed_files)} files failed to download:")
    for f in failed_files:
        print(f"  - {f}")
else:
    print("🎉 All files downloaded successfully!")
