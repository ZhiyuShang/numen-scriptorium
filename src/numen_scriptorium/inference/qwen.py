from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from numen_scriptorium.paths import ROOT


def _resolve_path(path_like: str) -> str:
    p = Path(path_like)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def load_model(base_model: str, lora_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, _resolve_path(lora_dir))
    model.eval()
    return tokenizer, model


def generate(
    tokenizer,
    model,
    instruction: str,
    user_input: str = "",
    max_new_tokens: int = 256,
    seed: int | None = None,
):
    if user_input:
        prompt = f"指令：{instruction}\n输入：{user_input}\n回答："
    else:
        prompt = f"指令：{instruction}\n回答："

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=inputs["input_ids"].device)
        generator.manual_seed(seed)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            generator=generator,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "回答：" in text:
        text = text.split("回答：", 1)[1]
    return text.strip()
