"""Backwards-compatible training entrypoint.

Old command kept working:
  python train_qlora_qwen.py --preset t4 --resume latest

New preferred command:
  python -m numen_scriptorium.cli.train --config configs/train_qwen_7b.yaml
"""

from numen_scriptorium.cli.train import main


if __name__ == "__main__":
    main()