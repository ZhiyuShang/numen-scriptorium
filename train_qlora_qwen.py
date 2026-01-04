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

BASE_MODEL = "Qwen/Qwen3-0.6B"   # Model name 
TRAIN_FILE = os.path.join("data", "train.jsonl")
OUTPUT_DIR = os.path.join("outputs", "qwen3_0_6b_boh_qlora")

MICRO_BATCH_SIZE = 1
GRADIENT_ACCUM_STEPS = 8
NUM_EPOCHS = 1.5          # Run 1~2 epochs first to see effects
LEARNING_RATE = 5e-5
MAX_SEQ_LEN = 768       

# =====================================================


def format_example(example: Dict) -> Dict:
    """
    Combine a {"instruction", "input", "output"} sample into a single text sequence.
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
    Encode dataset using tokenizer, and set labels to input_ids (standard SFT)
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
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 4bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 3. Load base model (4bit)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA Config (Qwen3 target_modules)
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

    # 6. Dataset
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE},
        split="train",
    )
    tokenized_train = encode_dataset(tokenizer, dataset)

    # 7. Training arguments
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
        save_steps=500,       # Overridden by save_strategy="epoch", but harmless to exist
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

    # 9. Train
    trainer.train()

    # 10. Save LoRA
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training complete! LoRA adapter and tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()