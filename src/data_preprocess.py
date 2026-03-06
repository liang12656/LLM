"""基础数据清洗脚本：用于合并并处理原始专利 CSV 数据。"""

from __future__ import annotations

import argparse
import glob
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_kernels

# 预编译不可见字符的正则，用于统一清洗
INVISIBLE_CHAR_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff\u00ad]")


def list_csv_files(pattern: str) -> List[Path]:
    """根据通配符列出所有 CSV 文件路径，兼容不同工作目录。"""

    def _glob(p: str) -> List[Path]:
        return [Path(found) for found in glob.glob(p)]

    paths = _glob(pattern)
    if not paths:
        # 当从 src/ 目录或其它位置运行脚本时，data/... 的相对路径可能失效。
        project_root = Path(__file__).resolve().parent.parent
        alt_pattern = str(project_root / Path(pattern))
        paths = _glob(alt_pattern)
        if paths:
            logging.info("未在当前工作目录找到文件，改用项目根路径解析 pattern=%s", alt_pattern)

    if not paths:
        raise FileNotFoundError(
            f"未找到匹配的 CSV 文件，pattern={pattern}，"
            "请确认路径是否相对项目根目录，或尝试使用绝对路径"
        )
    return sorted(paths)


def load_raw_data(csv_files: Iterable[Path]) -> pd.DataFrame:
    """读取多个 CSV，并在列对齐后进行合并。"""
    frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        logging.info("读取原始数据：%s", csv_file)
        frames.append(pd.read_csv(csv_file))
    merged = pd.concat(frames, ignore_index=True)
    logging.info("合并完成，总行数：%d", len(merged))
    return merged


def fullwidth_to_halfwidth(text: str) -> str:
    """将全角字符转换为半角字符。"""
    result_chars: List[str] = []
    for char in text:
        code = ord(char)
        if code == 0x3000:
            result_chars.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            result_chars.append(chr(code - 0xFEE0))
        else:
            result_chars.append(char)
    return "".join(result_chars)


def clean_text(text: str) -> str:
    """执行不可见字符清理、全角转半角、空白符压缩等操作。"""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = INVISIBLE_CHAR_PATTERN.sub("", text)
    text = fullwidth_to_halfwidth(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """完成列过滤、缺失值处理与文本清洗。"""
    required_cols = ["pub_id", "title", "abstract", "claims", "ipc"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必须列：{missing_cols}")

    df = df[required_cols].copy()
    df = df.dropna(subset=["pub_id", "abstract", "claims"])

    for col in ["pub_id", "title", "abstract", "claims", "ipc"]:
        logging.debug("清洗列：%s", col)
        df[col] = df[col].apply(clean_text)

    # 再次过滤空字符串（例如原始内容为空或清洗后为空）
    df = df[(df["pub_id"] != "") & (df["abstract"] != "") & (df["claims"] != "")]
    df.reset_index(drop=True, inplace=True)
    logging.info("清洗完成，剩余行数：%d", len(df))
    return df


def save_processed_data(df: pd.DataFrame, output_path: Path) -> Path:
    """保存处理后的数据，并确保输出目录存在。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("数据已保存至：%s", output_path)
    return output_path


def compute_similarity_candidates(
    df: pd.DataFrame,
    output_path: Path,
    top_k: int = 5,
    max_rows: Optional[int] = None,
    n_jobs: int = 1,
) -> Path:
    """根据摘要相似度为每条专利生成候选相似专利列表。"""

    if max_rows is not None and max_rows > 0:
        logging.info("仅对前 %d 条数据计算相似度 (demo 模式)", max_rows)
        work_df = df.head(max_rows).copy()
    else:
        work_df = df.copy()

    if len(work_df) <= 1:
        raise ValueError("样本量不足，无法计算相似专利")

    logging.info("开始进行 TF-IDF 向量化，样本数：%d", len(work_df))
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(work_df["abstract"].tolist())

    logging.info("计算 cosine 相似度，相似度矩阵尺寸：%s，n_jobs=%s", tfidf_matrix.shape, n_jobs)
    similarity_matrix = pairwise_kernels(tfidf_matrix, metric="cosine", n_jobs=n_jobs)
    np.fill_diagonal(similarity_matrix, -1.0)

    neighbor_records = []
    pub_ids = work_df["pub_id"].tolist()

    for idx, query_pub_id in enumerate(pub_ids):
        row = similarity_matrix[idx]
        effective_top_k = min(top_k, len(pub_ids) - 1)
        if effective_top_k <= 0:
            continue

        top_indices = np.argpartition(row, -effective_top_k)[-effective_top_k:]
        top_indices = top_indices[np.argsort(row[top_indices])[::-1]]

        for neighbor_idx in top_indices:
            neighbor_records.append(
                {
                    "query_pub_id": query_pub_id,
                    "neighbor_pub_id": pub_ids[neighbor_idx],
                    "similarity": float(row[neighbor_idx]),
                }
            )

    similarity_df = pd.DataFrame(neighbor_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    similarity_df.to_csv(output_path, index=False)
    logging.info("相似专利信息已保存至：%s", output_path)
    logging.info("TODO：可替换为 Sentence-BERT/FAISS 等更高质量向量检索方案")
    return output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="专利数据基础清洗")
    parser.add_argument(
        "--input_glob",
        default="data/raw/*.csv",
        help="原始 CSV 文件通配符，默认 data/raw/*.csv",
    )
    parser.add_argument(
        "--output",
        default="data/processed/all_patents_clean.csv",
        help="清洗后数据输出路径",
    )
    parser.add_argument(
        "--skip_similarity",
        action="store_true",
        help="是否跳过相似专利计算，默认执行",
    )
    parser.add_argument(
        "--similar_output",
        default="data/processed/similar_patents.csv",
        help="相似专利结果输出路径",
    )
    parser.add_argument(
        "--similar_top_k",
        type=int,
        default=5,
        help="每条专利保留的相似专利数量",
    )
    parser.add_argument(
        "--similar_max_rows",
        type=int,
        default=None,
        help="仅对前 N 条数据执行相似检索，便于大数据集 demo",
    )
    parser.add_argument(
        "--similar_n_jobs",
        type=int,
        default=1,
        help="相似度计算的并行度，-1 表示使用所有 CPU",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="日志级别，例如 INFO/DEBUG",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口：从读取、清洗到落盘的完整流程。"""
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")

    csv_files = list_csv_files(args.input_glob)
    raw_df = load_raw_data(csv_files)
    clean_df = clean_dataframe(raw_df)
    save_processed_data(clean_df, Path(args.output))

    if not args.skip_similarity:
        compute_similarity_candidates(
            clean_df,
            Path(args.similar_output),
            top_k=args.similar_top_k,
            max_rows=args.similar_max_rows,
            n_jobs=args.similar_n_jobs,
        )


if __name__ == "__main__":
    main()
