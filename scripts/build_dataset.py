"""Thin wrapper for dataset building script."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "build_train_data_from_raw.py"
    runpy.run_path(str(target), run_name="__main__")
