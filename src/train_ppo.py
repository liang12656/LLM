"""PPO/RLHF training script using TRL and a reward model."""

from dataclasses import dataclass
from typing import Any

from trl import PPOTrainer, PPOConfig


@dataclass
class PPOTrainingConfig:
    policy_model: str
    reward_model_path: str
    dataset_path: str
    output_dir: str = "checkpoints/ppo"


def load_ppo_dataset(dataset_path: str) -> Any:
    """Return prompts for PPO rollouts."""
    raise NotImplementedError


def create_ppo_trainer(config: PPOTrainingConfig) -> PPOTrainer:
    """Instantiate PPO trainer with policy/reference models and reward_fn."""
    raise NotImplementedError


def main(config_path: str | None = None) -> None:
    """Run PPO training loop and save adapters."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
