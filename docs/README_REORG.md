# Repository Reorganization Notes

This repository was reorganized to a more standard Python layout while preserving backward compatibility.

## New layout

- `src/numen_scriptorium/` – package code
  - `training/qlora.py` – training implementation
  - `inference/qwen.py` – inference implementation
  - `cli/` – CLI entrypoints (`train`, `infer`, `smoke`)
  - `common.py`, `config.py`, `paths.py` – shared helpers
- `configs/` – YAML training presets (`train_qwen_1_5b.yaml`, `train_qwen_7b.yaml`)
- `scripts/` – thin wrappers (`train.py`, `infer.py`)
- `docs/` – docs/migration notes

## Backward compatibility

Old entry files are still runnable and now delegate to package CLI:

- `train_qlora_qwen.py` -> `numen_scriptorium.cli.train`
- `infer_qlora_qwen3_boh.py` -> `numen_scriptorium.cli.infer`

## Recommended commands

- Train:
  - `python -m numen_scriptorium.cli.train --config configs/train_qwen_7b.yaml`
- Infer:
  - `python -m numen_scriptorium.cli.infer --adapter outputs/.../best`
- Smoke test:
  - `python -m numen_scriptorium.cli.smoke --config configs/train_qwen_7b.yaml`
