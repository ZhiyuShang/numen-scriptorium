"""Backwards-compatible inference entrypoint.

Old command kept working:
  python infer_qlora_qwen3_boh.py --adapter outputs/.../best

New preferred command:
  python -m numen_scriptorium.cli.infer --adapter outputs/.../best
"""

from numen_scriptorium.cli.infer import main


if __name__ == "__main__":
    main()