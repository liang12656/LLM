"""使用 LoRA 对 LLaMA2-7B 进行创造性分类 SFT。

本脚本读取 `data/processed/hf_dataset/` 下的 train/valid 数据集，
对 prompt 字段做分词，并加载 LLaMA 模型 + LoRA 适配层进行微调。
训练过程中会在验证集上计算 Accuracy / F1，并应用 EarlyStopping。

示例：
    python src/train_sft.py --dataset_dir data/processed/hf_dataset --output_dir checkpoints/sft_llama_creativity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="LoRA 微调创造性分类模型")
    parser.add_argument(
        "--dataset_dir",
        default="data/processed/hf_dataset",
        help="HF Dataset 保存目录（需包含 train/validation/test）",
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-2-7b-hf",
        help="预训练基座模型名称，可替换为可用的中文 LLaMA",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/sft_llama_creativity",
        help="LoRA Adapter 与 tokenizer 的保存路径",
    )
    parser.add_argument("--max_length", type=int, default=1024, help="输入序列最大长度")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="AdamW 学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减系数")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="单卡 batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="验证 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数，缓解显存压力")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="余弦学习率调度 warmup 比例")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志打印步数")
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=200,
        help="多少步执行一次验证与 EarlyStopping 检查",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="EarlyStopping 容忍次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA r 值 (秩)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="启用 FP16 训练（需硬件支持）。默认关闭使用 bfloat16/FP32",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="启用 BF16 训练（需硬件支持）。默认关闭使用 FP32",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="可选：仅取前 N 条训练样本进行调试",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="可选：仅取前 N 条验证样本进行调试",
    )
    return parser.parse_args()


def load_hf_dataset(dataset_dir: str) -> DatasetDict:
    """从磁盘加载 HuggingFace Dataset。"""

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} 不存在，请先运行 build_dataset.py")
    dataset = load_from_disk(str(dataset_path))
    if not {"train", "validation"}.issubset(dataset.keys()):
        raise ValueError("数据集需包含 train 与 validation split。")
    return dataset


def build_tokenizer(model_name: str) -> AutoTokenizer:
    """构建 tokenizer 并确保存在 pad token。"""

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # LLaMA 默认无 pad，手动补齐
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int, label_column: str = "creativity_label", sample_limits: Optional[Dict[str, int]] = None) -> DatasetDict:
    """对 prompt 字段分词并附带 label。"""

    def preprocess(examples: Dict[str, list[str]]) -> Dict[str, np.ndarray]:
        tokenized = tokenizer(
            examples["prompt"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )
        labels = examples[label_column]
        if isinstance(labels[0], str):
            labels = [int(l) for l in labels]
        tokenized["labels"] = labels
        return tokenized

    processed = dataset
    if sample_limits:
        for split, limit in sample_limits.items():
            if split in processed and limit is not None:
                processed[split] = processed[split].select(range(min(limit, len(processed[split]))))

    tokenized_dataset = processed.map(preprocess, batched=True, remove_columns=processed["train"].column_names)
    return tokenized_dataset


def prepare_model(model_name: str, tokenizer: AutoTokenizer, num_labels: int, lora_cfg: LoraConfig) -> AutoModelForSequenceClassification:
    """加载基座模型并挂载 LoRA 适配器。"""

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    # 若新增了 pad token，需调整模型 embedding 大小
    if tokenizer.pad_token_id is not None and model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    lora_cfg.task_type = TaskType.SEQ_CLS
    model = get_peft_model(model, lora_cfg)
    return model


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """计算 Accuracy / F1，用于 Trainer。"""

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_hf_dataset(args.dataset_dir)
    tokenizer = build_tokenizer(args.model_name)

    sample_limits = {
        "train": args.max_train_samples,
        "validation": args.max_eval_samples,
    }
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.max_length, sample_limits=sample_limits)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    model = prepare_model(args.model_name, tokenizer, num_labels=2, lora_cfg=lora_config)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=args.evaluation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.evaluation_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=["tensorboard"],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    eval_metrics = trainer.evaluate()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    # 保存最终指标，方便日志系统读取
    metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
    with open(Path(args.output_dir) / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("训练完成，最终验证指标：")
    print(json.dumps({k: float(v) for k, v in eval_metrics.items() if k.startswith("eval_")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
