import os
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ================= 用户需确认 / 修改的配置 =================

BASE_MODEL = "Qwen/Qwen3-0.6B"   # HF 上的模型名
TRAIN_FILE = os.path.join("data", "train.jsonl")
OUTPUT_DIR = os.path.join("outputs", "qwen3_0_6b_boh_qlora")

MICRO_BATCH_SIZE = 1
GRADIENT_ACCUM_STEPS = 8
NUM_EPOCHS = 1.5          # 先跑 1~2 轮看效果
LEARNING_RATE = 5e-5
MAX_SEQ_LEN = 768         # 你的段落有点长，768 比 512 安全

# =====================================================


def format_example(example: Dict) -> Dict:
    """
    将一条 {"instruction", "input", "output"} 样本拼成单条文本序列。
    """
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()

    if inp:
        prompt = f"指令：{instruction}\n输入：{inp}\n回答："
    else:
        prompt = f"指令：{instruction}\n回答："

    example["text"] = prompt + out
    return example


def encode_dataset(tokenizer, dataset):
    """
    使用 tokenizer 编码数据集，并把 labels 设为 input_ids（标准 SFT）
    """
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )

    dataset = dataset.map(format_example)
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized = tokenized.map(
        lambda x: {"labels": x["input_ids"]},
        batched=False,
    )
    return tokenized


def main():
    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 4bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 3. 加载基座模型（4bit）
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. 为 k-bit 训练做准备
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA Config（Qwen3 target_modules）
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. 数据集
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE},
        split="train",
    )
    tokenized_train = encode_dataset(tokenizer, dataset)

    # 7. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,

        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,

        do_eval=False,
        eval_strategy="no",

        logging_dir=None,
        logging_strategy="steps",
        logging_steps=10,

        save_strategy="epoch",
        save_steps=500,       # 被 save_strategy="epoch" 覆盖，但参数存在无害
        save_total_limit=2,

        fp16=False,
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        report_to=None,
        remove_unused_columns=False,
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
    )

    # 9. 训练
    trainer.train()

    # 10. 保存 LoRA
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"训练完成！LoRA 适配器和 tokenizer 已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()