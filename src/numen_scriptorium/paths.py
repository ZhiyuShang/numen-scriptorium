from pathlib import Path


def project_root() -> Path:
    current = Path(__file__).resolve()
    for p in [current, *current.parents]:
        if (p / "configs").exists() and (p / "data_split").exists():
            return p
    return Path.cwd()


ROOT = project_root()
