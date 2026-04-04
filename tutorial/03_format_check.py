# 03_format_check.py

# ===== import =====
from datasets import load_dataset

# ===== 本文 =====
DATASET_NAME = "yahma/alpaca-cleaned"
DATASET_SPLIT = "train[:5]"

print("=== Load Dataset ===")

dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

# カラム確認
print("Columns:", dataset.column_names)

required_keys = ["instruction", "input", "output"]

for key in required_keys:
    if key not in dataset.column_names:
        raise ValueError(f"Missing key: {key}")

print("Schema OK")

def format_func(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    return f"""Below is an instruction that describes a task.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""

# フォーマット実行
formatted = []
for i, sample in enumerate(dataset):
    try:
        text = format_func(sample)
        formatted.append(text)
    except Exception as e:
        print(f"Error at index {i}: {e}")

# ===== 結果 =====
print("=== Formatted Sample ===")
print(formatted[0])
print(f"Formatted count: {len(formatted)}")