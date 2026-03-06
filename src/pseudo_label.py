"""根据相似专利信息构造创造性伪标签。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd


def load_clean_patents(path: Path) -> pd.DataFrame:
    """读取已经清洗好的专利数据。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到清洗后的专利数据：{path}")
    df = pd.read_csv(path)
    required_cols = {"pub_id", "title", "abstract", "claims", "ipc"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"清洗数据缺少必要列：{missing}")
    logging.info("载入清洗数据，共 %d 条", len(df))
    return df


def load_similarity(path: Path) -> pd.DataFrame:
    """读取相似专利列表，用于生成伪标签。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到相似专利文件：{path}")
    df = pd.read_csv(path)
    required_cols = {"query_pub_id", "neighbor_pub_id", "similarity"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"相似专利文件缺少必要列：{missing}")
    logging.info("载入相似度数据，共 %d 条候选", len(df))
    return df


def assign_creativity_label(
    row: pd.Series,
    neighbors_df: pd.DataFrame,
    sim_threshold: float = 0.9,
    min_neighbors: int = 1,
    ipc_prefix_len: int = 4,
) -> int:
    """根据 IPC 前缀与高相似邻居数量打标，返回 0/1。"""

    if neighbors_df is None or neighbors_df.empty:
        return 1

    query_prefix = (row.get("ipc") or "")[:ipc_prefix_len]
    filtered = neighbors_df[
        (neighbors_df["similarity"] >= sim_threshold)
        & (neighbors_df["neighbor_ipc_prefix"] == query_prefix)
        & (neighbors_df["neighbor_ipc_prefix"] != "")
    ]
    return 0 if len(filtered) >= min_neighbors else 1


def build_neighbor_groups(
    neighbors: pd.DataFrame, ipc_map: Dict[str, str], ipc_prefix_len: int
) -> Dict[str, pd.DataFrame]:
    """为每个 query_pub_id 预先整理邻居列表，并附带 IPC 前缀。"""

    neighbors = neighbors.copy()
    neighbors["neighbor_ipc"] = neighbors["neighbor_pub_id"].map(ipc_map).fillna("")
    neighbors["neighbor_ipc_prefix"] = neighbors["neighbor_ipc"].str[:ipc_prefix_len]

    grouped: Dict[str, pd.DataFrame] = {}
    for pub_id, sub_df in neighbors.groupby("query_pub_id"):
        grouped[pub_id] = sub_df.reset_index(drop=True)
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据相似专利生成创造性伪标签")
    parser.add_argument(
        "--clean_data",
        default="data/processed/all_patents_clean.csv",
        help="清洗后的专利数据路径",
    )
    parser.add_argument(
        "--neighbors",
        default="data/processed/similar_patents.csv",
        help="相似专利列表路径",
    )
    parser.add_argument(
        "--output",
        default="data/processed/patents_with_labels.csv",
        help="带有伪标签的数据输出路径",
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.9,
        help="判定缺乏创造性所需的最低相似度阈值",
    )
    parser.add_argument(
        "--min_neighbors",
        type=int,
        default=1,
        help="达到阈值的最少邻居数量，满足则打 0 标签",
    )
    parser.add_argument(
        "--ipc_prefix_len",
        type=int,
        default=4,
        help="比较 IPC 是否一致时使用的前缀长度",
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

    clean_path = Path(args.clean_data)
    neighbors_path = Path(args.neighbors)
    output_path = Path(args.output)

    patents_df = load_clean_patents(clean_path)
    neighbors_df = load_similarity(neighbors_path)

    ipc_map = dict(zip(patents_df["pub_id"], patents_df["ipc"].fillna("")))
    neighbor_groups = build_neighbor_groups(neighbors_df, ipc_map, args.ipc_prefix_len)

    logging.info("开始根据规则生成创造性伪标签，TODO：可替换为更复杂的打分模型")
    patents_df["creativity_label"] = patents_df.apply(
        lambda row: assign_creativity_label(
            row,
            neighbor_groups.get(row["pub_id"]),
            sim_threshold=args.sim_threshold,
            min_neighbors=args.min_neighbors,
            ipc_prefix_len=args.ipc_prefix_len,
        ),
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    patents_df.to_csv(output_path, index=False)
    logging.info("伪标签结果已保存：%s", output_path)


if __name__ == "__main__":
    main()
