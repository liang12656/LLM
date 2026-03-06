# LLM

微调LLM_TAIDE_chat_demo_v1
此文件 不包含微调过程，只有微调前后的对比

## 专利创造性审查项目结构

该项目使用 Python 3.10+，并依赖 pandas、numpy、scikit-learn、torch、transformers、datasets、peft、trl、accelerate、sentencepiece 等库来完成从数据处理到 RLHF 的完整流程。目录结构如下：

```
.
├── checkpoints/               # 保存 SFT、奖励模型、PPO 权重
├── config/                    # 配置文件（YAML/JSON，可按需补充）
├── data/
│   ├── processed/             # 清洗、伪标注后的数据
│   └── raw/                   # 原始专利数据 CSV
├── logs/                      # 训练与评估日志
├── requirements.txt           # 依赖清单
└── src/
    ├── build_dataset.py       # 处理后的数据 -> HF Dataset
    ├── data_preprocess.py     # 数据清洗、伪标注
    ├── evaluate.py            # Accuracy / F1 评估脚本
    ├── inference.py           # 创造性审查推理入口
    ├── train_ppo.py           # PPO / RLHF 训练
    ├── train_reward_model.py  # 奖励模型训练
    └── train_sft.py           # LLaMA2-7B + LoRA 监督微调
```

### 脚本功能与结构概述

- **data_preprocess.py**：实现 `load_raw_data`、`clean_data`、`pseudo_label`、`save_processed_data` 等函数，`main` 用于串联原始数据加载、清洗、伪标注以及结果落盘。
- **build_dataset.py**：包含 `load_processed_files`、`build_hf_dataset`、`push_to_hub` 等函数，`main` 负责将处理后的数据构造成 HuggingFace `DatasetDict` 并可选上传。
- **train_sft.py**：定义 `SFTConfig` 数据类以及 `load_dataset`、`prepare_model`、`create_trainer`、`main` 等函数，完成 LLaMA2-7B + LoRA 的监督微调并输出检查点。
- **evaluate.py**：提供 `load_eval_data`、`run_inference`、`compute_metrics`、`main` 等函数，在验证/测试集上评估 Accuracy 与 F1。
- **train_reward_model.py**：定义 `RewardModelConfig`，通过 `load_pairwise_dataset`、`prepare_reward_model`、`main` 等函数训练奖励模型。
- **train_ppo.py**：包含 `PPOTrainingConfig`、`load_ppo_dataset`、`create_ppo_trainer`、`main`，用于基于奖励模型执行 PPO / RLHF。
- **inference.py**：通过 `InferenceConfig`、`load_model`、`review_creativity`、`main` 实现最终模型的创造性审查推理。

### 数据目录说明

- `data/raw/`：原始专利 CSV 或 JSON 数据，脚本 `data_preprocess.py` 从这里读取。
- `data/processed/`：清洗与伪标注后的数据，`build_dataset.py` 会从该目录构建训练/验证/测试集合。
