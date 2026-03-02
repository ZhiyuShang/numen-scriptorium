from __future__ import annotations

import argparse
import time

from numen_scriptorium.inference.qwen import generate, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--adapter",
        type=str,
        default="outputs/qwen2_5_1_5b_boh_qlora/checkpoint-1400",
        help="LoRA adapter path (default kept for legacy script compatibility).",
    )
    parser.add_argument("--instruction", type=str, default="请将输入翻译为中文。")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = load_model(args.base_model, args.adapter)
    out = generate(
        tokenizer,
        model,
        instruction=args.instruction,
        user_input=args.input,
        max_new_tokens=args.max_new_tokens,
    )
    print(out)
    print(f"Time taken: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
