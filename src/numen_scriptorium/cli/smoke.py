from __future__ import annotations

import argparse

from numen_scriptorium.config import apply_overrides, load_yaml_config
from numen_scriptorium.training.qlora import smoke_test_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_qwen_7b.yaml")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    cfg = apply_overrides(cfg, {"base_model": args.base_model})
    smoke_test_from_config(cfg, max_seq_len_override=args.max_seq_len)


if __name__ == "__main__":
    main()
