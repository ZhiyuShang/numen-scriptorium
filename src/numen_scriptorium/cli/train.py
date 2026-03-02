from __future__ import annotations

import argparse

from numen_scriptorium.config import apply_overrides, load_yaml_config
from numen_scriptorium.training.qlora import train_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_qwen_7b.yaml")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--preset", type=str, choices=["t4", "a100"], default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--report_to", type=str, choices=["wandb", "none"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic torch behavior (may reduce performance).",
    )
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    cfg = apply_overrides(
        cfg,
        {
            "base_model": args.base_model,
            "output_dir": args.output_dir,
            "preset": args.preset,
            "report_to": args.report_to,
            "seed": args.seed,
            "deterministic": args.deterministic if args.deterministic else None,
        },
    )
    train_from_config(cfg, resume=args.resume, max_seq_len_override=args.max_seq_len)


if __name__ == "__main__":
    main()
