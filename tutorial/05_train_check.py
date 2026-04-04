# 05_train_check.py

# ===== import =====
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ===== 本文 =====

MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024
DATASET_NAME = "yahma/alpaca-cleaned"
DATASET_SPLIT = "train[:100]"

print("=== STEP 1: Load Model ===")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# --- 検証① GPU確認 ---
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))

# --- 検証② LoRA前パラメータ ---
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

before_params = count_trainable_params(model)
print(f"Trainable params BEFORE LoRA: {before_params}")


print("=== STEP 2: Apply LoRA ===")

target_modules = [
    "q_proj","k_proj","v_proj","o_proj",
    "gate_proj","up_proj","down_proj",
]

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=target_modules,
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing=True,
)

after_params = count_trainable_params(model)
print(f"Trainable params AFTER LoRA: {after_params}")

if after_params >= before_params:
    raise ValueError("LoRA not applied correctly (params not reduced)")

# --- 検証③ trainable層確認 ---
trainable_layers = [n for n, p in model.named_parameters() if p.requires_grad]
if len(trainable_layers) == 0:
    raise ValueError("No trainable params")

print(f"Trainable layers: {len(trainable_layers)}")

print("=== STEP 3: Load Dataset ===")

dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

print("Columns:", dataset.column_names)

required_keys = ["instruction", "input", "output"]
for k in required_keys:
    if k not in dataset.column_names:
        raise ValueError(f"Missing key: {k}")

print(f"Dataset size: {len(dataset)}")

# --- 検証④ フォーマット ---
def format_func(example):
    return f"""Below is an instruction that describes a task.

### Instruction:
{example.get("instruction","")}

### Input:
{example.get("input","")}

### Response:
{example.get("output","")}"""

dataset = dataset.map(lambda x: {"text": format_func(x)})

# フォーマット確認
sample_text = dataset[0]["text"]
if len(sample_text) < 10:
    raise ValueError("Formatting failed")

print("Formatted sample OK")

print("=== STEP 4: Trainer Setup ===")

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=5,
    optim="adamw_8bit",
    output_dir="outputs",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

print("Trainer initialized")

print("=== STEP 5: Train ===")

train_result = trainer.train()

# --- 検証⑤ loss確認 ---
loss = train_result.training_loss
if loss is None:
    raise ValueError("Loss not returned")

print(f"Final loss: {loss}")

# --- 検証⑥ モデル更新確認 ---
updated_params = count_trainable_params(model)
if updated_params == before_params:
    raise ValueError("Model not updated")

print("Model updated confirmed")

# ===== 結果 =====
print("=== RESULT ===")
print("Training completed successfully")
print(f"Loss: {loss}")
print("Device:", torch.cuda.get_device_name(0))