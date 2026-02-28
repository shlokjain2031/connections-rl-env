from __future__ import annotations

from typing import Dict

from .env import CategoryLabelVisibility, OneAwayMode, RewardConfig, RewardMode


def get_preset_env_kwargs(name: str) -> Dict[str, object]:
    clean = name.strip().lower()

    if clean == "research_balanced":
        return {
            "action_mask_mode": "auto",
            "mask_strictness": "balanced",
            "one_away_mode": OneAwayMode.INFO_ONLY,
            "category_label_visibility": CategoryLabelVisibility.SOLVED_ONLY,
            "include_action_mask": True,
            "reward_config": RewardConfig(
                reward_mode=RewardMode.SHAPED,
                step_penalty=-0.01,
                solved_group_reward=1.0,
                wrong_group_penalty=-0.2,
                one_away_bonus=-0.05,
                invalid_action_penalty=-0.05,
                win_reward=2.0,
                lose_reward=-1.0,
            ),
        }

    if clean == "research_strict":
        return {
            "action_mask_mode": "auto",
            "mask_strictness": "strict",
            "one_away_mode": OneAwayMode.DISABLED,
            "category_label_visibility": CategoryLabelVisibility.NEVER,
            "include_action_mask": True,
            "reward_config": RewardConfig(
                reward_mode=RewardMode.SHAPED,
                step_penalty=-0.01,
                solved_group_reward=1.0,
                wrong_group_penalty=-0.25,
                one_away_bonus=-0.05,
                invalid_action_penalty=-0.05,
                win_reward=2.0,
                lose_reward=-1.0,
            ),
        }

    if clean == "benchmark_sparse":
        return {
            "action_mask_mode": "valid",
            "mask_strictness": "balanced",
            "one_away_mode": OneAwayMode.DISABLED,
            "category_label_visibility": CategoryLabelVisibility.NEVER,
            "include_action_mask": False,
            "reward_config": RewardConfig(
                reward_mode=RewardMode.SPARSE,
                win_reward=1.0,
                lose_reward=-1.0,
            ),
        }

    if clean == "benchmark_open":
        return {
            "action_mask_mode": "auto",
            "mask_strictness": "open",
            "one_away_mode": OneAwayMode.DISABLED,
            "category_label_visibility": CategoryLabelVisibility.NEVER,
            "include_action_mask": False,
            "reward_config": RewardConfig(
                reward_mode=RewardMode.SPARSE,
                win_reward=1.0,
                lose_reward=-1.0,
            ),
        }

    valid = ["research_balanced", "research_strict", "benchmark_sparse", "benchmark_open"]
    raise ValueError(f"Unknown preset: {name!r}. Expected one of: {', '.join(valid)}")
