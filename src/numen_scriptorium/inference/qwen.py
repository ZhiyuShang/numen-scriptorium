from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from numen_scriptorium.paths import ROOT


def _resolve_path(path_like: str) -> str:
    p = Path(path_like)
    if p.is_absolute() and p.exists():
        return str(p)

    candidate = ROOT / p
    if candidate.exists():
        return str(candidate)

    # Fall back to the original value (e.g. a Hugging Face repo id).
    return path_like


def load_model(base_model: str, lora_dir: str | None, use_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
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
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            base = base.to("cuda")

    model = base
    if lora_dir:
        model = PeftModel.from_pretrained(base, _resolve_path(lora_dir))

    model.eval()
    return tokenizer, model


def generate(
    tokenizer,
    model,
    instruction: str,
    user_input: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
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
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            generator=generator,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "回答：" in text:
        text = text.split("回答：", 1)[1]
    return text.strip()


def stream_generate(
    tokenizer,
    model,
    instruction: str,
    user_input: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    seed: int | None = None,
    stop_event: Event | None = None,
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

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    class _EventStoppingCriteria(StoppingCriteria):
        def __init__(self, event: Event):
            self._event = event

        def __call__(self, input_ids, scores, **kwargs):  # noqa: D401
            return self._event.is_set()

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        generator=generator,
        streamer=streamer,
    )
    if stop_event is not None:
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList([_EventStoppingCriteria(stop_event)])

    worker = Thread(target=model.generate, kwargs=generate_kwargs)
    worker.start()
    for new_text in streamer:
        if stop_event is not None and stop_event.is_set():
            break
        yield new_text
    worker.join(timeout=0.5)


def get_model_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        pass
    except Exception:
        pass
    return str(getattr(model, "device", "unknown"))
