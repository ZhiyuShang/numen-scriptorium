import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


def is_struct_example(example: Dict) -> bool:
    """
    Identify structured generation samples.
    These are the samples where instruction refers to structured item information.
    """
    instruction = (example.get("instruction") or "").lower()
    return (
        "structured item information" in instruction
        or "结构化信息" in instruction
        or "结构化" in instruction
    )


def read_jsonl(path: Path) -> List[Dict]:
    """Read a JSONL file into a list of dicts."""
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: Path, data: List[Dict]) -> None:
    """Write a list of dicts into a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def stratified_split(
    data: List[Dict],
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Perform stratified split based on whether a sample is structured or base.
    This keeps the structured/base ratio roughly consistent across splits.
    """
    random.seed(seed)

    structured = [x for x in data if is_struct_example(x)]
    base = [x for x in data if not is_struct_example(x)]

    random.shuffle(structured)
    random.shuffle(base)

    def split_bucket(bucket: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        n = len(bucket)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        test = bucket[:n_test]
        val = bucket[n_test:n_test + n_val]
        train = bucket[n_test + n_val:]
        return train, val, test

    train_s, val_s, test_s = split_bucket(structured)
    train_b, val_b, test_b = split_bucket(base)

    train = train_s + train_b
    val = val_s + val_b
    test = test_s + test_b

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True,
                        help="Input JSONL file (e.g., data/train_items_core.jsonl)")
    parser.add_argument("--out_dir", type=str, default="data_split",
                        help="Output directory for train/val/test JSONL files")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.0,
                        help="Test split ratio (0.0 if not needed)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    in_path = Path(args.in_file)
    out_dir = Path(args.out_dir)

    data = read_jsonl(in_path)
    assert len(data) > 0, f"No samples found in {in_path}"

    train, val, test = stratified_split(
        data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    if args.test_ratio > 0:
        write_jsonl(out_dir / "test.jsonl", test)

    print(f"Total samples: {len(data)}")
    print(f"Train: {len(train)} -> {out_dir / 'train.jsonl'}")
    print(f"Val:   {len(val)} -> {out_dir / 'val.jsonl'}")
    if args.test_ratio > 0:
        print(f"Test:  {len(test)} -> {out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    #example run in terminal:
    #python scripts/split_jsonl.py --in_file data/train_all.jsonl --val_ratio 0.05 --test_ratio 0.0 --seed 42
    main()