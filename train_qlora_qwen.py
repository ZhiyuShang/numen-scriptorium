import os
from typing import Dict
import argparse
import glob
import hashlib

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

# =======================
# Model / Paths
# =======================
# Better base model for instruction-following + translation/style tasks (8GB-friendly with 4bit QLoRA)
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

OUTPUT_DIR = os.path.join("outputs", "qwen2_5_1_5b_boh_qlora")

TRAIN_FILE = os.path.join("data_split", "train.jsonl")
VAL_FILE   = os.path.join("data_split", "val.jsonl")

# =======================
# Hyperparams (8GB VRAM safe defaults)
# =======================
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUM_STEPS = 16  # effective batch=16
NUM_EPOCHS = 4
LEARNING_RATE = 5e-5
MAX_SEQ_LEN = 512          # 8GB: start safe (you can try 768 later if it fits)

EVAL_STEPS = 200
SAVE_STEPS = 200
LOGGING_STEPS = 10

# =======================


def encode_dataset(tokenizer, dataset):
    """
    Encode dataset and apply prompt masking:
    - input_ids = prompt + answer
    - labels mask prompt tokens (-100), supervise only answer tokens
    """
    def build_and_tokenize(example):
        instruction = (example.get("instruction") or "").strip()
        inp = (example.get("input") or "").strip()
        out = (example.get("output") or "").strip()

        if inp:
            prompt = f"指令：{instruction}\n输入：{inp}\n回答："
        else:
            prompt = f"指令：{instruction}\n回答："

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(out, add_special_tokens=False)["input_ids"]

        # Append EOS to help generation stop
        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        # Truncate
        input_ids = input_ids[:MAX_SEQ_LEN]
        labels = labels[:MAX_SEQ_LEN]
        attention_mask = [1] * len(input_ids)

        # Pad to MAX_SEQ_LEN
        pad_id = tokenizer.pad_token_id
        pad_len = MAX_SEQ_LEN - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return dataset.map(build_and_tokenize, remove_columns=dataset.column_names)


def get_latest_checkpoint(output_dir: str):
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not ckpts:
        return None

    def step_num(p):
        name = os.path.basename(p)
        try:
            return int(name.split("-")[-1])
        except Exception:
            return -1

    ckpts = sorted(ckpts, key=step_num)
    return ckpts[-1]


def main(resume: str | None = None):
    # -----------------------
    # 1) Tokenizer
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )
    # Qwen models usually have eos; ensure pad exists for fixed-length padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------
    # 2) 4-bit quant config (8GB-friendly) + fp16 compute for 30-series
    # -----------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # IMPORTANT for RTX 30xx
    )

    # -----------------------
    # 3) Load model (4-bit)
    # -----------------------
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # -----------------------
    # 4) Memory savers for 8GB
    # -----------------------
    # Gradient checkpointing reduces VRAM usage notably (slower but safer)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # -----------------------
    # 5) LoRA config
    # -----------------------
    # r=16 is a safer default for 8GB; you can try r=32 if you have headroom.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------
    # 6) Dataset
    # -----------------------
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    )

    tokenized_train = encode_dataset(tokenizer, dataset["train"])
    tokenized_val   = encode_dataset(tokenizer, dataset["validation"])

    # -----------------------
    # 7) Training args (8GB-safe, more stable)
    # -----------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,

        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,

        do_eval=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=4,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=True,     # IMPORTANT for RTX 3070Ti
        bf16=False,

        report_to="wandb",
        run_name="qwen2_5_1_5b_boh_qlora_fp16",

        remove_unused_columns=False,
    )

    # -----------------------
    # 8) Trainer
    # -----------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

    # -----------------------
    # 9) Resume logic
    # -----------------------
    resume_from = None
    if resume is not None:
        if resume == "" or str(resume).lower() == "latest":
            resume_from = get_latest_checkpoint(OUTPUT_DIR)
        else:
            resume_from = resume

        if resume_from is None:
            print(f"[Resume] No checkpoint found in {OUTPUT_DIR}. Start from scratch.")
        else:
            print(f"[Resume] Resuming from: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    # -----------------------
    # 10) Save BEST LoRA (trainer.model is best when load_best_model_at_end=True)
    # -----------------------
    best_dir = os.path.join(OUTPUT_DIR, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    print("Best model saved to:", best_dir)
    print(f"Training complete! LoRA adapter and tokenizer saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume training from checkpoint. "
             "Use --resume (or --resume latest) to auto-pick latest, "
             "or --resume path/to/checkpoint-XXX to specify. "
             "Omit to start from scratch."
    )
    args = parser.parse_args()
    main(resume=args.resume)