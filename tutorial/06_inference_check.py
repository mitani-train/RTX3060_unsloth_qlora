# 06_inference_check.py

# ===== import =====
import torch
from unsloth import FastLanguageModel

# ===== 本文 =====

BASE_MODEL = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit"
LORA_PATH = "outputs/checkpoint-13"
MAX_NEW_TOKENS = 200

print("=== STEP 1: Load Base Model ===")

model, tokenizer = FastLanguageModel.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
)

# --- GPU確認 ---
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available")

device_name = torch.cuda.get_device_name(0)
print("GPU:", device_name)

print("=== STEP 2: Load LoRA ===")
# STEP 2: Load LoRA
from peft import PeftModel

model = PeftModel.from_pretrained(model, LORA_PATH)

print("LoRA loaded")

# --- 推論モード ---
FastLanguageModel.for_inference(model)

print("=== STEP 3: Prepare Prompt ===")

prompt = """Below is an instruction that describes a task.

### Instruction:
Explain QLoRA simply.

### Response:
"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

print("Tokenization OK")

print("=== STEP 4: Generate ===")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
    )

decoded = tokenizer.decode(outputs[0])

print("=== RESULT ===")
print(decoded)
print("=== INFO ===")
print("Device:", device_name)