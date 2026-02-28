"""Connections reinforcement learning environment."""

from .env import (
    ActionMaskMode,
    CategoryLabelVisibility,
    ConnectionsEnv,
    MaskStrictness,
    OneAwayMode,
    RewardConfig,
    RewardMode,
)
from .presets import get_preset_env_kwargs
from .puzzles import Puzzle, PuzzleBank, load_default_puzzle_bank, make_puzzle_bank
from .registration import register_envs
from .vector_env import ConnectionsVectorEnv

try:
    from .wrappers import ActionMaskToInfoWrapper, FlattenConnectionsObservation
except (ModuleNotFoundError, ImportError):  # pragma: no cover - wrappers need gymnasium
    ActionMaskToInfoWrapper = None  # type: ignore[assignment]
    FlattenConnectionsObservation = None  # type: ignore[assignment]

__all__ = [
    "ActionMaskMode",
    "MaskStrictness",
    "OneAwayMode",
    "CategoryLabelVisibility",
    "ConnectionsEnv",
    "RewardConfig",
    "RewardMode",
    "ConnectionsVectorEnv",
    "get_preset_env_kwargs",
    "Puzzle",
    "PuzzleBank",
    "load_default_puzzle_bank",
    "make_puzzle_bank",
    "register_envs",
    "ActionMaskToInfoWrapper",
    "FlattenConnectionsObservation",
]

register_envs()
