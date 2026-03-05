from __future__ import annotations

import os
import queue
import re
import threading
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr

from numen_scriptorium.inference.qwen import get_model_device, load_model, stream_generate


BASE_MODEL = os.getenv("NS_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER = os.getenv("NS_ADAPTER", "ICGenAIShare06/boh-qlora-adapter/best").strip() or None
USE_4BIT = os.getenv("NS_USE_4BIT", "1") == "1"
DEFAULT_INSTRUCTION = os.getenv("NS_DEFAULT_INSTRUCTION", "请将输入翻译为中文，并保持原文风格。")

_RUNTIME_LOADED = False
_ACTIVE_STOP_EVENT: threading.Event | None = None
_STOP_LOCK = threading.Lock()


def _set_active_stop_event(stop_event: threading.Event | None):
    global _ACTIVE_STOP_EVENT
    with _STOP_LOCK:
        _ACTIVE_STOP_EVENT = stop_event


def _request_stop():
    with _STOP_LOCK:
        if _ACTIVE_STOP_EVENT is not None:
            _ACTIVE_STOP_EVENT.set()


def _on_stop_clicked():
    _request_stop()
    return _format_status(
        stage="Stop requested",
        loaded=_RUNTIME_LOADED,
        device="unknown",
        loading_percent="--",
        error="Stop requested. Waiting for backend generation to halt.",
    )


def _on_clear_clicked():
    # Clear should also stop any in-flight generation to avoid concurrent
    # updates from the stream generator after UI has been reset.
    _request_stop()
    return (
        DEFAULT_INSTRUCTION,
        "",
        "",
        _format_status(stage="Idle", loaded=_RUNTIME_LOADED, device="unknown", loading_percent="0%"),
        "0.00s",
    )


def _format_loading_percent(value: int) -> str:
    return f"{max(0, min(100, int(value)))}%"


def _infer_example_label(instruction: str, user_input: str, idx: int) -> str:
    lower_instruction = instruction.lower()
    if "sun's design" in user_input.lower():
        return "BoH EN→ZH (Sun's Design)"
    if "velvet lesson" in user_input.lower() or "moth and dream" in lower_instruction:
        return "Moth&Dream EN→ZH (Velvet Lesson)"
    if "deposition" in lower_instruction:
        return "EN Generation (Deposition)"
    if "generate one entry" in lower_instruction or "catalog" in lower_instruction:
        return "EN Generation (Catalog Entry)"
    return f"Example {idx + 1}"


def _load_demo_examples():
    """Load examples from demo_examples.txt / demo_example.txt.

    Expected per block:
    - python infer_qlora_qwen3_boh.py ...
    - --instruction "..."
    - --input "..."
    - optional --max_new_tokens <int>
    """
    candidate_files = [
        Path(__file__).resolve().parent / "demo_examples.txt",
        Path(__file__).resolve().parent / "demo_example.txt",
    ]
    file_path = next((p for p in candidate_files if p.exists()), None)
    if file_path is None:
        return [], "⚠️ Examples file not found (expected demo_examples.txt)."

    try:
        raw = file_path.read_text(encoding="utf-8")
    except Exception:
        return [], "⚠️ Could not read examples file."

    block_pattern = re.compile(
        r"python\s+infer_qlora_qwen3_boh\.py(?P<body>.*?)(?=(?:\n\s*python\s+infer_qlora_qwen3_boh\.py)|\Z)",
        re.DOTALL,
    )
    instruction_pattern = re.compile(r'--instruction\s+"(?P<instruction>.*?)"\s*`', re.DOTALL)
    input_pattern = re.compile(r'--input\s+"(?P<input>.*?)"\s*`', re.DOTALL)
    max_tokens_pattern = re.compile(r"--max_new_tokens\s+(?P<max_new_tokens>\d+)")

    parsed = []
    for idx, block in enumerate(block_pattern.finditer(raw)):
        body = block.group("body")
        instruction_match = instruction_pattern.search(body)
        input_match = input_pattern.search(body)
        if not instruction_match or not input_match:
            continue

        instruction = instruction_match.group("instruction").strip()
        user_input = input_match.group("input").strip()
        max_match = max_tokens_pattern.search(body)
        max_new_tokens = int(max_match.group("max_new_tokens")) if max_match else None

        parsed.append(
            {
                "label": _infer_example_label(instruction, user_input, idx),
                "instruction": instruction,
                "input": user_input,
                "max_new_tokens": max_new_tokens,
            }
        )

    if not parsed:
        return [], "⚠️ Failed to parse demo examples. Please check examples file format."
    return parsed, None


def _apply_example(example: dict):
    max_tokens_update = (
        example["max_new_tokens"] if example.get("max_new_tokens") is not None else gr.update()
    )
    return example["instruction"], example["input"], max_tokens_update


def _format_status(
    *,
    stage: str,
    loaded: bool,
    device: str,
    loading_percent: str | None = None,
    elapsed: float | None = None,
    error: str | None = None,
    stream_chunks: int | None = None,
    output_chars: int | None = None,
):
    lines = [
        "### Model / System status",
        f"- **Stage:** {stage}",
        f"- **Model loaded:** {'✅ Yes' if loaded else '❌ No'}",
        f"- **Device:** `{device}`",
        f"- **Base model:** `{BASE_MODEL}`",
        f"- **Adapter:** `{ADAPTER or 'None'}`",
        f"- **4-bit quantization:** `{USE_4BIT}`",
    ]
    if loading_percent is not None:
        lines.append(f"- **Model loading:** `{loading_percent}`")
    if elapsed is not None:
        lines.append(f"- **Time per request:** `{elapsed:.2f}s`")
    if stream_chunks is not None:
        lines.append(f"- **Stream chunks received:** `{stream_chunks}`")
    if output_chars is not None:
        lines.append(f"- **Output characters so far:** `{output_chars}`")
    if error:
        lines.append(f"- **Error:** ⚠️ {error}")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_runtime():
    global _RUNTIME_LOADED
    runtime = load_model(base_model=BASE_MODEL, lora_dir=ADAPTER, use_4bit=USE_4BIT)
    _RUNTIME_LOADED = True
    return runtime


def run_inference_stream(
    instruction: str,
    user_input: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    start = time.perf_counter()
    device = "unknown"
    stage = "Preparing request"
    load_progress = 0
    cleaned_instruction = instruction.strip() or DEFAULT_INSTRUCTION
    cleaned_input = user_input.strip()
    stop_event = threading.Event()
    _set_active_stop_event(stop_event)

    if not cleaned_input:
        msg = "⚠️ Please provide input text before running generation."
        yield (
            msg,
            _format_status(
                stage="Waiting for input",
                loaded=_RUNTIME_LOADED,
                device=device,
                loading_percent=_format_loading_percent(load_progress),
            ),
            "0.00s",
        )
        _set_active_stop_event(None)
        return

    try:
        stage = "Loading model"
        if _RUNTIME_LOADED:
            tokenizer, model = get_runtime()
            load_progress = 100
            yield (
                "",
                _format_status(
                    stage="Model ready (cached)",
                    loaded=True,
                    device=device,
                    loading_percent=_format_loading_percent(load_progress),
                ),
                f"{time.perf_counter() - start:.2f}s",
            )
        else:
            runtime_box: dict[str, tuple] = {}
            err_box: dict[str, Exception] = {}

            def _loader():
                try:
                    runtime_box["runtime"] = get_runtime()
                except Exception as exc:
                    err_box["error"] = exc

            loader_thread = threading.Thread(target=_loader, daemon=True)
            loader_thread.start()

            load_progress = 3
            while loader_thread.is_alive():
                if stop_event.is_set():
                    elapsed = time.perf_counter() - start
                    yield (
                        "⚠️ Stop requested. Model loading may continue in background.",
                        _format_status(
                            stage="Stopped during model loading",
                            loaded=False,
                            device=device,
                            loading_percent=_format_loading_percent(load_progress),
                            elapsed=elapsed,
                        ),
                        f"{elapsed:.2f}s",
                    )
                    return

                load_progress = min(95, load_progress + 4)
                elapsed = time.perf_counter() - start
                yield (
                    "",
                    _format_status(
                        stage=f"Loading model ({load_progress}%)",
                        loaded=False,
                        device=device,
                        loading_percent=_format_loading_percent(load_progress),
                        elapsed=elapsed,
                    ),
                    f"{elapsed:.2f}s",
                )
                time.sleep(0.2)

            loader_thread.join()
            if "error" in err_box:
                raise err_box["error"]
            tokenizer, model = runtime_box["runtime"]
            load_progress = 100

        device = get_model_device(model)

        stage = "Tokenizing / preparing generation"
        elapsed = time.perf_counter() - start
        yield (
            "",
            _format_status(
                stage=stage,
                loaded=True,
                device=device,
                loading_percent=_format_loading_percent(load_progress),
                elapsed=elapsed,
                stream_chunks=0,
                output_chars=0,
            ),
            f"{elapsed:.2f}s",
        )

        stage = "Generating"
        partial = ""
        chunk_count = 0
        token_queue: queue.Queue[str | None] = queue.Queue()
        error_queue: queue.Queue[Exception] = queue.Queue()

        def _token_producer():
            try:
                for token in stream_generate(
                    tokenizer=tokenizer,
                    model=model,
                    instruction=cleaned_instruction,
                    user_input=cleaned_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    stop_event=stop_event,
                ):
                    token_queue.put(token)
            except Exception as exc:
                error_queue.put(exc)
            finally:
                token_queue.put(None)

        producer = threading.Thread(target=_token_producer, daemon=True)
        producer.start()

        first_token_seen = False
        while True:
            if stop_event.is_set():
                elapsed = time.perf_counter() - start
                yield (
                    partial.strip(),
                    _format_status(
                        stage="Stopped by user",
                        loaded=True,
                        device=device,
                        loading_percent=_format_loading_percent(load_progress),
                        elapsed=elapsed,
                        stream_chunks=chunk_count,
                        output_chars=len(partial.strip()),
                    ),
                    f"{elapsed:.2f}s",
                )
                return

            if not error_queue.empty():
                raise error_queue.get()

            try:
                delta = token_queue.get(timeout=0.2)
            except queue.Empty:
                elapsed = time.perf_counter() - start
                wait_stage = "Generating (waiting for first token)" if not first_token_seen else "Generating"
                yield (
                    partial,
                    _format_status(
                        stage=wait_stage,
                        loaded=True,
                        device=device,
                        loading_percent=_format_loading_percent(load_progress),
                        elapsed=elapsed,
                        stream_chunks=chunk_count,
                        output_chars=len(partial),
                    ),
                    f"{elapsed:.2f}s",
                )
                continue

            if delta is None:
                break

            first_token_seen = True
            chunk_count += 1
            partial += delta
            elapsed = time.perf_counter() - start
            yield (
                partial,
                _format_status(
                    stage=stage,
                    loaded=True,
                    device=device,
                    loading_percent=_format_loading_percent(load_progress),
                    elapsed=elapsed,
                    stream_chunks=chunk_count,
                    output_chars=len(partial),
                ),
                f"{elapsed:.2f}s",
            )

        elapsed = time.perf_counter() - start
        yield (
            partial.strip(),
            _format_status(
                stage="Done",
                loaded=True,
                device=device,
                loading_percent=_format_loading_percent(load_progress),
                elapsed=elapsed,
                stream_chunks=chunk_count,
                output_chars=len(partial.strip()),
            ),
            f"{elapsed:.2f}s",
        )
    except Exception:
        elapsed = time.perf_counter() - start
        friendly = "Generation failed. Please check model / adapter settings and try again."
        yield (
            f"⚠️ {friendly}",
            _format_status(
                stage=stage,
                loaded=_RUNTIME_LOADED,
                device=device,
                loading_percent=_format_loading_percent(load_progress),
                elapsed=elapsed,
                error=friendly,
            ),
            f"{elapsed:.2f}s",
        )
    finally:
        _set_active_stop_event(None)


with gr.Blocks(title="Numen Scriptorium Demo") as demo:
    gr.Markdown("# ✨ Numen Scriptorium · HF Demo")
    gr.Markdown(
        "This demo can: (1) translate EN↔ZH with Book-of-Hours/Cultist-Simulator-like tone., and (2) rewrite/generate text with instructed tone and nouns.\n\n"
        "For lore-like quality, load a matching LoRA adapter (base model alone is not enough).\n\n"
        "**How to use**\n"
        "1. Keep or edit the instruction.\n"
        "2. Paste your input text.\n"
        "3. Click **Run** to generate output."
    )

    with gr.Row():
        with gr.Column(scale=3):
            instruction = gr.Textbox(label="Instruction", value=DEFAULT_INSTRUCTION, lines=3)
            user_input = gr.Textbox(label="Input", placeholder="在这里输入待翻译/待改写文本", lines=8)

            with gr.Accordion("Advanced settings", open=False):
                max_new_tokens = gr.Slider(32, 1024, value=256, step=16, label="max_new_tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")

            gr.Markdown("### Examples")
            gr.Markdown("Click an example button to auto-fill Instruction and Input.")
            parsed_examples, example_warning = _load_demo_examples()
            if example_warning:
                gr.Markdown(example_warning)

            with gr.Row():
                for example in parsed_examples:
                    example_btn = gr.Button(example["label"], variant="secondary")
                    example_btn.click(
                        fn=lambda ex=example: _apply_example(ex),
                        inputs=None,
                        outputs=[instruction, user_input, max_new_tokens],
                    )

            with gr.Row():
                run_btn = gr.Button("Run", variant="primary")
                stop_btn = gr.Button("Stop")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=2):
            output = gr.Markdown(label="Output", value="")
            elapsed_text = gr.Textbox(label="Elapsed", value="0.00s", interactive=False)
            status_panel = gr.Markdown(
                _format_status(stage="Idle", loaded=False, device="unknown", loading_percent="0%"),
                label="Model / System status",
            )

    run_event = run_btn.click(
        fn=run_inference_stream,
        inputs=[instruction, user_input, max_new_tokens, temperature, top_p],
        outputs=[output, status_panel, elapsed_text],
    )

    stop_btn.click(fn=_on_stop_clicked, inputs=None, outputs=[status_panel], cancels=[run_event])

    clear_btn.click(
        fn=_on_clear_clicked,
        inputs=None,
        outputs=[instruction, user_input, output, status_panel, elapsed_text],
        cancels=[run_event],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
