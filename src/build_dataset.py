"""将伪标注结果转换为 HuggingFace ``Dataset`` 并切分训练/验证/测试。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

from datasets import Dataset, DatasetDict, load_dataset

PROMPT_TEMPLATE = (
    "你是一名专利审查员，请根据给定的专利信息判断该专利是否具有创造性，并给出简要理由。\n"
    "专利标题：{title}\n"
    "专利摘要：{abstract}\n"
    "权利要求：{claims}\n"
    "请回答：该专利是否具有创造性？请输出：\n"
    "标签：0 或 1\n"
    "理由：一句话说明理由。"
)

REQUIRED_COLUMNS = {"pub_id", "title", "abstract", "claims", "ipc", "creativity_label"}


def load_labeled_dataset(csv_path: Path) -> Dataset:
    """使用 datasets 库读取 CSV 并校验必要列。"""

    if not csv_path.exists():
        raise FileNotFoundError(f"未找到带伪标签的数据：{csv_path}")

    dataset = load_dataset("csv", data_files={"all": str(csv_path)})["all"]
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}")
    return dataset


def format_prompt_fields(example: Dict[str, str]) -> Dict[str, str]:
    """为每条样本生成 prompt/target/full_answer 字段。"""

    label = str(example["creativity_label"])
    prompt = PROMPT_TEMPLATE.format(
        title=example.get("title", ""),
        abstract=example.get("abstract", ""),
        claims=example.get("claims", ""),
    )
    full_answer = f"标签：{label}\n理由：该标签基于启发式伪标注规则。"
    return {"prompt": prompt, "target": label, "full_answer": full_answer}


def train_valid_test_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """按照给定比例划分 train/valid/test，并返回 DatasetDict。"""

    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/valid/test 比例之和必须为 1")

    train_test_split = dataset.train_test_split(test_size=1 - train_ratio, seed=seed)
    train_dataset = train_test_split["train"]
    remainder = train_test_split["test"]

    if valid_ratio == 0 or test_ratio == 0:
        return DatasetDict({"train": train_dataset, "test": remainder})

    valid_size = valid_ratio / (valid_ratio + test_ratio)
    valid_test_split = remainder.train_test_split(test_size=1 - valid_size, seed=seed)
    return DatasetDict(
        {
            "train": train_dataset,
            "valid": valid_test_split["train"],
            "test": valid_test_split["test"],
        }
    )


def add_prompt_columns(dataset: Dataset) -> Dataset:
    """在 HF Dataset 上新增 prompt/target/full_answer 字段。"""

    return dataset.map(format_prompt_fields, desc="构建 prompt 字段")


def save_dataset_dict(datasets_dict: DatasetDict, output_dir: Path, save_format: str = "arrow") -> None:
    """将切分后的数据集保存到指定目录，支持 arrow(save_to_disk) 或 jsonl。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_format == "arrow":
        datasets_dict.save_to_disk(str(output_dir))
    elif save_format == "jsonl":
        for split, split_dataset in datasets_dict.items():
            target_path = output_dir / f"{split}.jsonl"
            split_dataset.to_json(str(target_path))
    else:
        raise ValueError("save_format 仅支持 'arrow' 或 'jsonl'")


def build_hf_dataset(
    csv_path: Path,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """从 CSV 构建包含 prompt/target 的 DatasetDict。"""

    dataset = load_labeled_dataset(csv_path)
    dataset = add_prompt_columns(dataset)
    return train_valid_test_split(dataset, train_ratio, valid_ratio, test_ratio, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 HuggingFace Dataset")
    parser.add_argument(
        "--input",
        default="data/processed/patents_with_labels.csv",
        help="伪标注 CSV 路径",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/hf_dataset",
        help="HF Dataset 保存目录",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_format",
        choices=["arrow", "jsonl"],
        default="arrow",
        help="输出格式：arrow(save_to_disk) 或 jsonl",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="日志级别，例如 INFO/DEBUG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")

    csv_path = Path(args.input)
    output_dir = Path(args.output_dir)

    datasets_dict = build_hf_dataset(
        csv_path,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    save_dataset_dict(datasets_dict, output_dir, save_format=args.save_format)
    logging.info("HF Dataset 已保存至：%s", output_dir)


if __name__ == "__main__":
    main()
