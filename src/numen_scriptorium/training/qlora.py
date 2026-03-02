from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Any, Dict

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

from numen_scriptorium.paths import ROOT


PRESET_CONFIGS = {
    "t4": {
        "max_seq_len": 1024,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "fp16": True,
        "bf16": False,
    },
    "a100": {
        "max_seq_len": 2048,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "lora_r": 32,
        "lora_alpha": 64,
        "learning_rate": 1e-4,
        "fp16": False,
        "bf16": True,
    },
}

COMMON_QWEN_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]


def _resolve_path(path_like: str | Path) -> str:
    p = Path(path_like)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def encode_dataset(tokenizer, dataset, max_seq_len: int):
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

        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        attention_mask = [1] * len(input_ids)

        pad_id = tokenizer.pad_token_id
        pad_len = max_seq_len - len(input_ids)
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


def get_lora_target_modules(model):
    found = set()
    for name, _ in model.named_modules():
        for candidate in COMMON_QWEN_TARGET_MODULES:
            if name == candidate or name.endswith(f".{candidate}"):
                found.add(candidate)
    selected = [m for m in COMMON_QWEN_TARGET_MODULES if m in found]
    if not selected:
        raise ValueError("No expected LoRA target modules found.")
    return selected


def pick_compute_dtype(use_bf16: bool):
    if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_training_config(config: Dict[str, Any], max_seq_len_override: int | None):
    preset = config.get("preset", "a100")
    cfg = dict(PRESET_CONFIGS[preset])
    cfg.update({k: v for k, v in config.items() if v is not None})

    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_vram_gb < 40:
            cfg["max_seq_len"] = min(int(cfg["max_seq_len"]), 1024)
            cfg["micro_batch_size"] = min(int(cfg["micro_batch_size"]), 1)
            cfg["gradient_accumulation_steps"] = max(int(cfg["gradient_accumulation_steps"]), 16)

    if max_seq_len_override is not None:
        cfg["max_seq_len"] = max_seq_len_override
    return cfg


def train_from_config(config: Dict[str, Any], resume: str | None = None, max_seq_len_override: int | None = None):
    cfg = resolve_training_config(config, max_seq_len_override)

    base_model = cfg.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    output_dir = _resolve_path(cfg.get("output_dir", "outputs/qwen2_5_7b_boh_qlora"))
    train_file = _resolve_path(cfg.get("train_file", "data_split/train.jsonl"))
    val_file = _resolve_path(cfg.get("val_file", "data_split/val.jsonl"))
    report_to = cfg.get("report_to", "wandb")

    use_bf16 = bool(cfg.get("bf16", False) and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = not use_bf16
    compute_dtype = pick_compute_dtype(use_bf16)
    report_to_value = [] if str(report_to).lower() == "none" else report_to

    model_name = base_model.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    run_name = cfg.get("run_name") or f"{model_name}_{cfg['preset']}_qlora"

    print("=" * 80)
    print("Effective training config")
    print(f"base_model: {base_model}")
    print(f"preset: {cfg['preset']}")
    print(f"max_seq_len: {cfg['max_seq_len']}")
    print(f"micro_batch_size: {cfg['micro_batch_size']}")
    print(f"gradient_accumulation_steps: {cfg['gradient_accumulation_steps']}")
    print(f"learning_rate: {cfg['learning_rate']}")
    print(f"lora(r/alpha): {cfg['lora_r']}/{cfg['lora_alpha']}")
    print(f"fp16: {use_fp16}, bf16: {use_bf16}")
    print(f"output_dir: {output_dir}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_target_modules = get_lora_target_modules(model)
    print(f"LoRA target modules: {lora_target_modules}")

    lora_config = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    tokenized_train = encode_dataset(tokenizer, dataset["train"], int(cfg["max_seq_len"]))
    tokenized_val = encode_dataset(tokenizer, dataset["validation"], int(cfg["max_seq_len"]))

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(cfg["micro_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(cfg.get("num_epochs", 4)),
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        do_eval=True,
        eval_strategy="steps",
        eval_steps=int(cfg.get("eval_steps", 200)),
        logging_strategy="steps",
        logging_steps=int(cfg.get("logging_steps", 10)),
        logging_first_step=True,
        save_strategy="steps",
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=int(cfg.get("save_total_limit", 4)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=report_to_value,
        run_name=run_name,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

    resume_from = None
    if resume is not None:
        if resume == "" or str(resume).lower() == "latest":
            resume_from = get_latest_checkpoint(output_dir)
        else:
            resume_from = resume
        if resume_from is None:
            print(f"[Resume] No checkpoint found in {output_dir}. Start from scratch.")
        else:
            print(f"[Resume] Resuming from: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print("Best model saved to:", best_dir)


def smoke_test_from_config(config: Dict[str, Any], max_seq_len_override: int | None = None):
    cfg = resolve_training_config(config, max_seq_len_override)
    base_model = cfg.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    train_file = _resolve_path(cfg.get("train_file", "data_split/train.jsonl"))

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    ds_all = load_dataset("json", data_files={"train": train_file})["train"]
    n = min(2, len(ds_all))
    ds = ds_all.select(range(n))
    tok = encode_dataset(tokenizer, ds, int(cfg["max_seq_len"]))
    batch = {
        "input_ids": torch.tensor([tok[i]["input_ids"] for i in range(n)], device=model.device),
        "attention_mask": torch.tensor([tok[i]["attention_mask"] for i in range(n)], device=model.device),
    }
    with torch.no_grad():
        _ = model(**batch)
    print(f"Smoke test passed: tokenizer/model loaded and {n} samples forward-passed.")
