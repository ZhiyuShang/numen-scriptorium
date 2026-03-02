"""Backwards-compatible stub. Use scripts/cs_extractor.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "cs_extractor.py"), run_name="__main__")
