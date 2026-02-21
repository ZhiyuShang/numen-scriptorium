import os
from typing import Dict
import wandb

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
OUTPUT_DIR = os.path.join("outputs", "qwen3_0_6b_boh_qlora")

MICRO_BATCH_SIZE = 1
GRADIENT_ACCUM_STEPS = 16
NUM_EPOCHS = 3         # Run 1~2 epochs first to see effects
LEARNING_RATE = 8e-5  
MAX_SEQ_LEN = 768       

TRAIN_FILE = os.path.join("data_split", "train.jsonl")
VAL_FILE   = os.path.join("data_split", "val.jsonl")

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
    Encode dataset and apply prompt masking:
    - Input_ids contain: prompt + answer
    - Labels ignore the prompt part (set to -100), and supervise only the answer tokens
    """
    def build_and_tokenize(example):
        instruction = (example.get("instruction") or "").strip()
        inp = (example.get("input") or "").strip()
        out = (example.get("output") or "").strip()

        if inp:
            prompt = f"指令：{instruction}\n输入：{inp}\n回答："
        else:
            prompt = f"指令：{instruction}\n回答："

        # Tokenize prompt and answer separately (no padding here)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(out, add_special_tokens=False)["input_ids"]

        # Optionally append EOS to the answer (helps generation stop)
        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

        # Concatenate
        input_ids = prompt_ids + answer_ids

        # Create labels with prompt masked out
        labels = [-100] * len(prompt_ids) + answer_ids

        # Truncate to max length
        input_ids = input_ids[:MAX_SEQ_LEN]
        labels = labels[:MAX_SEQ_LEN]

        # Build attention mask
        attention_mask = [1] * len(input_ids)

        # Pad to max length
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

    # Map to model-ready features
    tokenized = dataset.map(
        build_and_tokenize,
        remove_columns=dataset.column_names,
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
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj",],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
        },
    )

    tokenized_train = encode_dataset(tokenizer, dataset["train"])
    tokenized_val   = encode_dataset(tokenizer, dataset["validation"])

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,

        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        max_grad_norm=1.0,

        do_eval=True,
        eval_strategy="steps"
        eval_steps=100,

        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,

        save_strategy="epoch",
        save_total_limit=2,

        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),

        report_to="wandb",  # enable wandb logging
        run_name="qwen3_0_6b_boh_qlora_exp1",
        remove_unused_columns=False,
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

    # 9. Train
    trainer.train()

    # 10. Save LoRA
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training complete! LoRA adapter and tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()