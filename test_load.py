from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = 1024,  # 最初は下げる（重要）
    load_in_4bit = True,
)