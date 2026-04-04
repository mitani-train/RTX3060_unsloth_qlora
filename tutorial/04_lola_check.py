# 04_lora_check.py

# ===== import =====
import torch
from unsloth import FastLanguageModel

# ===== 本文 =====
MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024

print("=== Load Model ===")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# --- 前提検証①：モデル構造 ---
print("=== Check Modules ===")

target_modules = [
    "q_proj","k_proj","v_proj","o_proj",
    "gate_proj","up_proj","down_proj",
]

model_modules = dict(model.named_modules())
missing = [m for m in target_modules if not any(m in name for name in model_modules)]

if missing:
    raise ValueError(f"Target modules not found: {missing}")

print("All target modules found")

# --- 前提検証②：パラメータ数（適用前） ---
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

before_params = count_trainable_params(model)
print(f"Trainable params BEFORE LoRA: {before_params}")

# --- LoRA適用 ---
print("=== Apply LoRA ===")

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=target_modules,
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing=True,
)

# --- 前提検証③：パラメータ数（適用後） ---
after_params = count_trainable_params(model)
print(f"Trainable params AFTER LoRA: {after_params}")

if after_params >= before_params:
    raise ValueError("LoRA not applied correctly (params not reduced)")

# --- 前提検証④：requires_grad確認 ---
trainable_layers = [
    name for name, p in model.named_parameters() if p.requires_grad
]

if len(trainable_layers) == 0:
    raise ValueError("No trainable parameters found after LoRA")

# ===== 結果 =====
print("=== RESULT ===")
print("LoRA applied successfully")
print(f"Trainable layers count: {len(trainable_layers)}")
print(f"Example layer: {trainable_layers[0]}")
print("Device:", torch.cuda.get_device_name(0))