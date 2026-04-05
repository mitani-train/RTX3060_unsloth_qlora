# compare_inference_final.py

import torch
from unsloth import FastLanguageModel
from peft import PeftModel
import gc
import sys
import yaml
import os

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



config_path = sys.argv[1]
CONFIG = load_config(config_path)
BASE_MODEL = CONFIG["model"]["name"]

PROMPT = """Below is an instruction that describes a task.

### Instruction:
LLM学習のLoRAについてシンプルに解説してください。

### Response:
"""

def generate(model, tokenizer):
    inputs = tokenizer([PROMPT], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
        )

    result = tokenizer.decode(outputs[0])

    # 明示解放
    del inputs, outputs
    torch.cuda.empty_cache()

    return result


def main():

    print("GPU:", torch.cuda.get_device_name(0))

    # ===== BASE =====
    print("\n=== BASE MODEL ===")

    model, tokenizer = FastLanguageModel.from_pretrained(
        BASE_MODEL,
        load_in_4bit=CONFIG["QLoRA"]
    )

    FastLanguageModel.for_inference(model)

    base_output = generate(model, tokenizer)
    print(base_output)

    # ===== 完全解放 =====
    del model, tokenizer, base_output
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[INFO] VRAM cleared")

    # ===== LORA =====
    print("\n=== LORA MODEL ===")

    model, tokenizer = FastLanguageModel.from_pretrained(
        BASE_MODEL,
        load_in_4bit=CONFIG["QLoRA"],
    )
    save_path = os.path.join(CONFIG["output"]["dir"], "final")

    model = PeftModel.from_pretrained(model, save_path)

    FastLanguageModel.for_inference(model)

    lora_output = generate(model, tokenizer)
    print(lora_output)


if __name__ == "__main__":
    main()