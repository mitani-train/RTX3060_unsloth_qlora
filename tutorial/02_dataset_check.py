# 02_dataset_check.py

# ===== import =====
from datasets import load_dataset

# ===== 本文 =====
DATASET_NAME = "yahma/alpaca-cleaned"
DATASET_SPLIT = "train[:100]"

print("=== Load Dataset ===")

dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

# ===== 結果 =====
print("Dataset loaded")
print("Size:", len(dataset))
print("Sample:", dataset[0])