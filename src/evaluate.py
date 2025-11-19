"""在测试集上评估 LoRA 微调后的 LLaMA 创造性分类模型。

示例：
    python src/evaluate.py \
        --model_dir checkpoints/sft_llama_creativity \
        --base_model meta-llama/Llama-2-7b-hf \
        --dataset_dir data/processed/hf_dataset --split test
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 LoRA 微调后的创造性分类模型")
    parser.add_argument(
        "--model_dir",
        default="checkpoints/sft_llama_creativity",
        help="保存 LoRA Adapter 与 tokenizer 的目录",
    )
    parser.add_argument(
        "--base_model",
        default="meta-llama/Llama-2-7b-hf",
        help="与微调阶段一致的基座模型名称",
    )
    parser.add_argument(
        "--dataset_dir",
        default="data/processed/hf_dataset",
        help="构建好的 HuggingFace Dataset 保存目录",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="需要评估的数据切分名称，例如 test/validation",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="评估 batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="prompt 最长长度")
    parser.add_argument(
        "--pred_output",
        default="data/processed/test_predictions.csv",
        help="保存预测结果的 CSV 路径",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="日志级别，例如 INFO/DEBUG",
    )
    return parser.parse_args()


def load_dataset_split(dataset_dir: Path, split: str):
    """加载指定切分的数据集，并确保包含 prompt 与 creativity_label。"""

    if not dataset_dir.exists():
        raise FileNotFoundError(f"数据集目录 {dataset_dir} 不存在，请先运行 build_dataset.py")

    datasets_dict = load_from_disk(str(dataset_dir))
    if split not in datasets_dict:
        raise ValueError(f"数据集中不存在 {split} 切分，可用切分：{list(datasets_dict.keys())}")

    required_cols = {"prompt", "creativity_label"}
    missing = required_cols - set(datasets_dict[split].column_names)
    if missing:
        raise ValueError(f"评估切分缺少必要列：{missing}")

    return datasets_dict[split]


def prepare_tokenizer(model_dir: Path, base_model: str) -> AutoTokenizer:
    """优先从微调目录加载 tokenizer，如不存在则回退到基座模型。"""

    source = model_dir if (model_dir / "tokenizer_config.json").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"
    return tokenizer


def load_lora_model(model_dir: Path, base_model: str, tokenizer: AutoTokenizer) -> PeftModel:
    """加载基座模型并挂载 LoRA Adapter。"""

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    if tokenizer.pad_token_id is not None and base.get_input_embeddings().num_embeddings != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer))

    peft_model = PeftModel.from_pretrained(base, str(model_dir))
    peft_model.eval()
    return peft_model


def tokenize_dataset(dataset, tokenizer: AutoTokenizer, max_length: int):
    """对 prompt 字段做编码，并将标签转换为 int。"""

    def preprocess(examples: Dict[str, List[str]]):
        encodings = tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        labels = [int(label) for label in examples["creativity_label"]]
        encodings["labels"] = labels
        return encodings

    processed = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    processed.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return processed


def run_evaluation(model: PeftModel, dataloader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    """运行推理，返回预测与真实标签。"""

    preds: List[int] = []
    labels: List[int] = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())

    return {"predictions": np.array(preds), "labels": np.array(labels)}


def compute_metrics(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """计算整体 Accuracy/Precision/Recall/F1 以及负类指标。"""

    preds = results["predictions"]
    labels = results["labels"]
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_label0": precision_score(labels, preds, pos_label=0, zero_division=0),
        "recall_label0": recall_score(labels, preds, pos_label=0, zero_division=0),
        "f1_label0": f1_score(labels, preds, pos_label=0, zero_division=0),
    }
    return metrics


def save_predictions(dataset, preds: np.ndarray, output_path: Path) -> None:
    """将预测与真实标签保存为 CSV，方便误差分析。"""

    records = {
        "pub_id": dataset["pub_id"] if "pub_id" in dataset.column_names else list(range(len(preds))),
        "prompt": dataset["prompt"],
        "label": [int(v) for v in dataset["creativity_label"]],
        "prediction": preds.tolist(),
    }
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    split_dataset = load_dataset_split(dataset_dir, args.split)

    tokenizer = prepare_tokenizer(model_dir, args.base_model)
    tokenized_dataset = tokenize_dataset(split_dataset, tokenizer, args.max_length)

    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lora_model(model_dir, args.base_model, tokenizer).to(device)

    results = run_evaluation(model, dataloader, device)
    metrics = compute_metrics(results)

    logging.info("评估指标：%s", metrics)

    report = classification_report(
        results["labels"],
        results["predictions"],
        digits=4,
        target_names=["label_0", "label_1"],
        zero_division=0,
    )
    print("\nClassification Report:\n" + report)

    save_predictions(split_dataset, results["predictions"], Path(args.pred_output))
    logging.info("预测结果已保存至 %s", args.pred_output)


if __name__ == "__main__":
    main()
