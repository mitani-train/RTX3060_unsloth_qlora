# 01_load_model.py

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

# ===== 結果 =====
print("Model loaded")
print("Device:", torch.cuda.get_device_name(0))
print("Vocab size:", tokenizer.vocab_size)