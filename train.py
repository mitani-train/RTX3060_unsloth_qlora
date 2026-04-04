# train.py

# ===== import =====
import unsloth

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import yaml
import sys

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===== util =====

def log(msg):
    print(f"[LOG] {msg}")

def format_prompt(example):
    return f"""以下はタスクを説明する指示です。

### 指示:
{example.get("instruction", example.get("question", ""))}

### 入力:
{example.get("input", "")}

### 回答:
{example.get("output", example.get("answer", ""))}
"""

# ===== main =====

def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python train.py configs/xxx.yaml")

    config_path = sys.argv[1]
    CONFIG = load_config(config_path)
    

    log(f"Using config: {config_path}")

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA not available")

    log(f"GPU: {torch.cuda.get_device_name(0)}")

    # ===== モデルロード =====
    log("Loading model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model"]["name"],
        max_seq_length=CONFIG["model"]["max_seq_length"],
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== LoRA適用 =====
    log("Applying LoRA...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora"]["r"],
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
        ],
        lora_alpha=CONFIG["lora"]["alpha"],
        lora_dropout=CONFIG["lora"]["dropout"],
        use_gradient_checkpointing=True,
    )

    # ===== データ読み込み =====
    log("Loading dataset...")

    dataset = load_dataset(
        CONFIG["dataset"]["name"],
        split=CONFIG["dataset"]["split"],
    )

    log(f"Dataset size: {len(dataset)}")

    # ===== 前処理 =====
    log("Formatting dataset...")

    dataset = dataset.map(
        lambda x: {"text": format_prompt(x)},
        remove_columns=dataset.column_names,
    )

    # ===== Training設定 =====
    log("Setting training args...")

    use_bf16 = torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        per_device_train_batch_size=CONFIG["training"]["batch_size"],
        gradient_accumulation_steps=CONFIG["training"]["grad_accum"],
        num_train_epochs=CONFIG["training"]["epochs"],
        learning_rate=CONFIG["training"]["lr"],
        output_dir=CONFIG["output"]["dir"],
    )

    # ===== Trainer =====
    log("Initializing trainer...")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=lambda x: x["text"],
        processing_class=tokenizer,   # ← これ追加（重要）
        args=training_args,
    )

    # ===== 学習 =====
    log("Training start...")

    result = trainer.train()

    log(f"Final loss: {result.training_loss}")

    # ===== 保存 =====
    log("Saving model...")

    save_path = os.path.join(CONFIG["output"]["dir"], "final")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    log(f"Saved to: {save_path}")
    log("Training completed")

# ===== entry =====

if __name__ == "__main__":
    main()