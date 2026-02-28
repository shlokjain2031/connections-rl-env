from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .puzzles import Puzzle, PuzzleBank, load_default_puzzle_bank

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYMNASIUM = True
except ImportError:  # pragma: no cover
    gym = object  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _HAS_GYMNASIUM = False


class ActionMaskMode(str, Enum):
    ALL = "all"
    VALID = "valid"
    CONSISTENT = "consistent"
    AUTO = "auto"


class MaskStrictness(str, Enum):
    STRICT = "strict"
    BALANCED = "balanced"
    OPEN = "open"


class OneAwayMode(str, Enum):
    DISABLED = "disabled"
    INFO_ONLY = "info_only"
    OBSERVATION = "observation"
    REWARD_SHAPING = "reward_shaping"


class CategoryLabelVisibility(str, Enum):
    NEVER = "never"
    SOLVED_ONLY = "solved_only"
    ALWAYS = "always"


class RewardMode(str, Enum):
    SPARSE = "sparse"
    SHAPED = "shaped"


@dataclass(frozen=True)
class RewardConfig:
    reward_mode: RewardMode = RewardMode.SPARSE
    step_penalty: float = -0.01
    solved_group_reward: float = 1.0
    wrong_group_penalty: float = -0.2
    one_away_bonus: float = -0.05
    invalid_action_penalty: float = -0.05
    win_reward: float = 2.0
    lose_reward: float = -1.0


class ConnectionsEnv(gym.Env if _HAS_GYMNASIUM else object):  # type: ignore[misc]
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    WORD_PAD_TOKEN = 26
    LABEL_PAD_TOKEN = 26
    LABEL_UNKNOWN_TOKEN = 27

    def __init__(
        self,
        *,
        puzzle_bank: Optional[PuzzleBank] = None,
        mistake_budget: int = 4,
        max_steps: Optional[int] = 32,
        max_invalid_actions: int = 16,
        reward_config: Optional[RewardConfig] = None,
        include_action_mask: bool = True,
        action_mask_mode: Union[str, ActionMaskMode] = "auto",
        mask_strictness: Union[str, MaskStrictness] = "balanced",
        one_away_mode: Union[str, OneAwayMode] = "disabled",
        category_label_visibility: Union[str, CategoryLabelVisibility] = "never",
        max_word_length: int = 12,
        max_label_length: int = 16,
        render_mode: Optional[str] = None,
        include_info: bool = True,
    ) -> None:
        if mistake_budget <= 0:
            raise ValueError("mistake_budget must be >= 1")
        if max_invalid_actions < 0:
            raise ValueError("max_invalid_actions must be >= 0")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be > 0 when provided")
        if max_word_length <= 0:
            raise ValueError("max_word_length must be >= 1")
        if max_label_length <= 0:
            raise ValueError("max_label_length must be >= 1")

        self.mistake_budget = int(mistake_budget)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.max_invalid_actions = int(max_invalid_actions)
        self.reward_config = reward_config or RewardConfig()
        self.include_action_mask = bool(include_action_mask)
        self._action_mask_mode = self._parse_mask_mode(action_mask_mode)
        self._mask_strictness = self._parse_mask_strictness(mask_strictness)
        self._one_away_mode = self._parse_one_away_mode(one_away_mode)
        self._category_label_visibility = self._parse_category_label_visibility(category_label_visibility)
        self.max_word_length = int(max_word_length)
        self.max_label_length = int(max_label_length)
        self.render_mode = render_mode
        self.include_info = bool(include_info)
        if self.render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be one of: None, 'ansi'")

        self.puzzle_bank = puzzle_bank or load_default_puzzle_bank()
        self.dataset_hash_full = self.puzzle_bank.dataset_hash_full
        self.dataset_hash_short = self.dataset_hash_full[:8]
        self.puzzles = list(self.puzzle_bank.puzzles)
        if not self.puzzles:
            raise ValueError("Puzzle bank must contain at least one puzzle.")

        self._all_actions = np.array(list(combinations(range(16), 4)), dtype=np.int16)
        self._action_lookup = {tuple(action.tolist()): idx for idx, action in enumerate(self._all_actions)}
        self._mask_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

        self._rng = Random()
        self._puzzle: Optional[Puzzle] = None
        self._attempts_used = 0
        self._mistakes_used = 0
        self._invalid_actions_used = 0
        self._done = False

        self._word_solved_mask = np.zeros(16, dtype=np.uint8)
        self._group_solved_mask = np.zeros(4, dtype=np.uint8)
        self._group_by_word = np.full(16, fill_value=-1, dtype=np.int8)
        self._words_encoded = np.full((16, self.max_word_length), fill_value=self.WORD_PAD_TOKEN, dtype=np.uint8)
        self._labels_encoded = np.full(
            (4, self.max_label_length),
            fill_value=self.LABEL_UNKNOWN_TOKEN,
            dtype=np.uint8,
        )
        self._true_labels_encoded = np.full(
            (4, self.max_label_length),
            fill_value=self.LABEL_PAD_TOKEN,
            dtype=np.uint8,
        )
        self._last_one_away = 0

        if _HAS_GYMNASIUM:
            self.action_space = spaces.Discrete(len(self._all_actions))  # type: ignore[attr-defined]
            obs_spaces: Dict[str, object] = {
                "words": spaces.Box(
                    low=0,
                    high=self.WORD_PAD_TOKEN,
                    shape=(16, self.max_word_length),
                    dtype=np.uint8,
                ),
                "active_word_mask": spaces.Box(low=0, high=1, shape=(16,), dtype=np.uint8),
                "word_solved_mask": spaces.Box(low=0, high=1, shape=(16,), dtype=np.uint8),
                "group_solved_mask": spaces.Box(low=0, high=1, shape=(4,), dtype=np.uint8),
                "group_labels": spaces.Box(
                    low=0,
                    high=self.LABEL_UNKNOWN_TOKEN,
                    shape=(4, self.max_label_length),
                    dtype=np.uint8,
                ),
                "attempts_used": spaces.Box(low=0, high=1_000_000, shape=(1,), dtype=np.int32),
                "mistakes_used": spaces.Box(low=0, high=self.mistake_budget, shape=(1,), dtype=np.int32),
                "invalid_actions_used": spaces.Box(
                    low=0,
                    high=max(1, self.max_invalid_actions if self.max_invalid_actions > 0 else 1_000_000),
                    shape=(1,),
                    dtype=np.int32,
                ),
                "groups_solved_count": spaces.Box(low=0, high=4, shape=(1,), dtype=np.int32),
                "words_solved_count": spaces.Box(low=0, high=16, shape=(1,), dtype=np.int32),
            }
            if self._one_away_mode == OneAwayMode.OBSERVATION:
                obs_spaces["one_away"] = spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8)
            if self.include_action_mask:
                obs_spaces["action_mask"] = spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(self._all_actions),),
                    dtype=np.uint8,
                )
            self.observation_space = spaces.Dict(obs_spaces)  # type: ignore[attr-defined]

    @property
    def attempts_used(self) -> int:
        return self._attempts_used

    @property
    def mistakes_used(self) -> int:
        return self._mistakes_used

    @property
    def invalid_actions_used(self) -> int:
        return self._invalid_actions_used

    @property
    def current_puzzle_id(self) -> str:
        if self._puzzle is None:
            return ""
        return self._puzzle.puzzle_id

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, object]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        if _HAS_GYMNASIUM:
            super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        selected: Optional[Puzzle] = None
        if options is not None and "puzzle_id" in options:
            puzzle_id = str(options["puzzle_id"]).strip()
            for puzzle in self.puzzles:
                if puzzle.puzzle_id == puzzle_id:
                    selected = puzzle
                    break
            if selected is None:
                raise ValueError(f"Unknown puzzle_id: {puzzle_id}")

        if selected is None:
            selected = self.puzzles[self._rand_index(len(self.puzzles))]

        self._puzzle = selected
        self._attempts_used = 0
        self._mistakes_used = 0
        self._invalid_actions_used = 0
        self._done = False
        self._word_solved_mask.fill(0)
        self._group_solved_mask.fill(0)
        self._group_by_word.fill(-1)
        self._words_encoded.fill(self.WORD_PAD_TOKEN)
        self._labels_encoded.fill(self.LABEL_UNKNOWN_TOKEN)
        self._true_labels_encoded.fill(self.LABEL_PAD_TOKEN)
        self._last_one_away = 0
        self._mask_cache.clear()

        for group_idx, group in enumerate(selected.groups):
            for word_idx in group:
                self._group_by_word[word_idx] = group_idx

        for idx, word in enumerate(selected.words):
            self._words_encoded[idx] = self._encode_word(word)
        for idx, label in enumerate(selected.labels):
            self._true_labels_encoded[idx] = self._encode_label(label)

        self._refresh_visible_labels()

        obs = self._build_observation()
        info = {
            "puzzle_id": selected.puzzle_id,
            "dataset_hash": self.dataset_hash_short,
            "dataset_hash_full": self.dataset_hash_full,
            "attempts_used": self._attempts_used,
            "mistakes_used": self._mistakes_used,
            "invalid_actions_used": self._invalid_actions_used,
            "transition_reason": "reset",
        }
        return obs, info

    def step(
        self,
        action: Union[int, np.integer, Sequence[int]],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, object]]:
        if self._puzzle is None:
            raise RuntimeError("Environment must be reset before stepping.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        selected_indices, invalid_reason = self._resolve_action(action)
        if invalid_reason is not None or selected_indices is None:
            self._invalid_actions_used += 1
            reward = float(self.reward_config.invalid_action_penalty)
            truncated = (
                self.max_invalid_actions > 0 and self._invalid_actions_used >= self.max_invalid_actions
            )
            if truncated:
                self._done = True
            self._last_one_away = 0
            action_mask_u8 = self._current_mask_u8() if self.include_action_mask else None
            obs = self._build_observation(action_mask_u8=action_mask_u8)
            reason = "invalid_truncation" if truncated else "invalid_action"
            info = self._make_info(
                transition_reason=reason,
                selected_indices=() if selected_indices is None else selected_indices,
                is_valid_action=False,
                invalid_reason=invalid_reason,
                one_away=False,
            )
            return obs, reward, False, truncated, info

        is_correct, solved_group_idx = self._evaluate_selection(selected_indices)
        one_away = False if is_correct else self._is_one_away(selected_indices)
        self._last_one_away = int(one_away)
        self._attempts_used += 1

        solved_puzzle = False
        mistakes_exhausted = False
        transition_reason = ""

        if is_correct and solved_group_idx is not None:
            self._group_solved_mask[solved_group_idx] = 1
            for word_idx in self._puzzle.groups[solved_group_idx]:
                self._word_solved_mask[word_idx] = 1
            self._refresh_visible_labels()
            solved_puzzle = bool(self._group_solved_mask.sum() == 4)
            transition_reason = "puzzle_solved" if solved_puzzle else "solved_group"
        else:
            self._mistakes_used += 1
            mistakes_exhausted = self._mistakes_used >= self.mistake_budget
            transition_reason = "mistakes_exhausted" if mistakes_exhausted else "wrong_group"

        terminated = solved_puzzle or mistakes_exhausted

        truncated = False
        if not terminated and self.max_steps is not None and self._attempts_used >= self.max_steps:
            truncated = True
            transition_reason = "max_steps_truncation"

        self._done = terminated or truncated
        reward = self._compute_reward(
            is_correct=is_correct,
            one_away=one_away,
            solved_puzzle=solved_puzzle,
            mistakes_exhausted=mistakes_exhausted,
        )
        action_mask_u8 = self._current_mask_u8() if self.include_action_mask else None
        obs = self._build_observation(action_mask_u8=action_mask_u8)
        info = self._make_info(
            transition_reason=transition_reason,
            selected_indices=selected_indices,
            is_valid_action=True,
            solved_group_index=solved_group_idx,
            invalid_reason=None,
            one_away=one_away,
        )
        return obs, reward, terminated, truncated, info

    def render(self) -> str:
        if self._puzzle is None:
            return ""
        chunks: List[str] = [f"Puzzle: {self._puzzle.puzzle_id}"]
        for group_idx, group in enumerate(self._puzzle.groups):
            solved = self._group_solved_mask[group_idx] == 1
            prefix = "[SOLVED]" if solved else "[     ]"
            words = ", ".join(self._puzzle.words[i] for i in group)
            chunks.append(f"{prefix} G{group_idx}: {words}")
        chunks.append(
            f"attempts={self._attempts_used} mistakes={self._mistakes_used}/{self.mistake_budget} invalid={self._invalid_actions_used}"
        )
        return "\n".join(chunks)

    def action_to_indices(self, action_index: int) -> Tuple[int, int, int, int]:
        if action_index < 0 or action_index >= len(self._all_actions):
            raise ValueError(f"Action index out of range: {action_index}")
        return tuple(int(v) for v in self._all_actions[action_index])

    def indices_to_action(self, indices: Sequence[int]) -> int:
        if len(indices) != 4:
            raise ValueError("indices must contain exactly 4 values")
        clean = tuple(sorted(int(v) for v in indices))
        if len(set(clean)) != 4:
            raise ValueError("indices must be unique")
        for idx in clean:
            if idx < 0 or idx >= 16:
                raise ValueError(f"word index out of range: {idx}")
        if clean not in self._action_lookup:
            raise ValueError("indices do not correspond to a valid 4-subset action")
        return self._action_lookup[clean]

    def valid_action_mask(self) -> np.ndarray:
        entry = self._current_mask_entry()
        # Immutability assumption: cached masks are treated read-only.
        return entry["mask"]

    def sample_valid_action(self) -> int:
        entry = self._current_mask_entry()
        valid_indices = entry["valid_indices"]
        if valid_indices.size == 0:
            return self._rand_index(len(self._all_actions))
        return int(valid_indices[self._rand_index(valid_indices.size)])

    def _build_observation(
        self,
        *,
        action_mask_u8: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        active_word_mask = (self._word_solved_mask == 0).astype(np.uint8)
        obs: Dict[str, np.ndarray] = {
            "words": self._words_encoded.copy(),
            "active_word_mask": active_word_mask,
            "word_solved_mask": self._word_solved_mask.copy(),
            "group_solved_mask": self._group_solved_mask.copy(),
            "group_labels": self._labels_encoded.copy(),
            "attempts_used": np.array([self._attempts_used], dtype=np.int32),
            "mistakes_used": np.array([self._mistakes_used], dtype=np.int32),
            "invalid_actions_used": np.array([self._invalid_actions_used], dtype=np.int32),
            "groups_solved_count": np.array([int(self._group_solved_mask.sum())], dtype=np.int32),
            "words_solved_count": np.array([int(self._word_solved_mask.sum())], dtype=np.int32),
        }
        if self._one_away_mode == OneAwayMode.OBSERVATION:
            obs["one_away"] = np.array([self._last_one_away], dtype=np.uint8)
        if self.include_action_mask:
            mask_u8 = self._current_mask_u8() if action_mask_u8 is None else action_mask_u8
            obs["action_mask"] = mask_u8.copy()
        return obs

    def _current_mask_entry(self) -> Dict[str, np.ndarray]:
        mode = self._effective_mask_mode()
        cache_key = (mode.value, self._get_solved_groups_bitmask())
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached

        if mode == ActionMaskMode.ALL:
            mask = np.ones(len(self._all_actions), dtype=np.bool_)
        else:
            # Stage 2: CONSISTENT intentionally aliases VALID until logical
            # hypothesis pruning is implemented.
            active = self._word_solved_mask == 0
            mask = np.all(active[self._all_actions], axis=1)

        valid_indices = np.flatnonzero(mask)
        mask_u8 = mask.view(np.uint8)
        self._validate_mask(mask)
        self._validate_valid_indices(valid_indices)
        self._validate_mask_u8(mask_u8)
        entry = {"mask": mask, "valid_indices": valid_indices, "mask_u8": mask_u8}
        self._mask_cache[cache_key] = entry
        return entry

    def _current_mask_u8(self) -> np.ndarray:
        return self._current_mask_entry()["mask_u8"]

    def _get_solved_groups_bitmask(self) -> int:
        bitmask = 0
        for group_idx in range(4):
            if self._group_solved_mask[group_idx] == 1:
                bitmask |= 1 << group_idx
        return bitmask

    def _validate_mask(self, mask: np.ndarray) -> None:
        assert mask.shape == (len(self._all_actions),)
        assert mask.dtype == np.bool_

    def _validate_valid_indices(self, valid_indices: np.ndarray) -> None:
        assert valid_indices.ndim == 1
        assert valid_indices.dtype.kind in ("i", "u")

    def _validate_mask_u8(self, mask_u8: np.ndarray) -> None:
        assert mask_u8.shape == (len(self._all_actions),)
        assert mask_u8.dtype == np.uint8

    def _make_info(
        self,
        *,
        transition_reason: str,
        selected_indices: Sequence[int],
        is_valid_action: bool,
        solved_group_index: Optional[int] = None,
        invalid_reason: Optional[str] = None,
        one_away: Optional[bool] = None,
    ) -> Dict[str, object]:
        if not self.include_info:
            return {}
        info: Dict[str, object] = {
            "puzzle_id": self.current_puzzle_id,
            "dataset_hash": self.dataset_hash_short,
            "dataset_hash_full": self.dataset_hash_full,
            "attempts_used": self._attempts_used,
            "mistakes_used": self._mistakes_used,
            "invalid_actions_used": self._invalid_actions_used,
            "transition_reason": transition_reason,
            "selected_word_indices": list(selected_indices),
            "is_valid_action": is_valid_action,
            "solved_group_index": solved_group_index,
            "invalid_reason": invalid_reason,
            "effective_action_mask_mode": self._effective_mask_mode().value,
        }
        if self._one_away_mode != OneAwayMode.DISABLED:
            info["one_away"] = bool(one_away) if one_away is not None else False
        return info

    def _evaluate_selection(self, indices: Sequence[int]) -> Tuple[bool, Optional[int]]:
        assert self._puzzle is not None
        i0, i1, i2, i3 = (int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3]))
        g0 = int(self._group_by_word[i0])
        if g0 < 0 or g0 >= 4:
            return False, None
        if int(self._group_by_word[i1]) != g0:
            return False, None
        if int(self._group_by_word[i2]) != g0:
            return False, None
        if int(self._group_by_word[i3]) != g0:
            return False, None

        group_id = g0

        true_group = self._puzzle.groups[group_id]
        if (
            self._group_solved_mask[group_id] == 0
            and i0 in true_group
            and i1 in true_group
            and i2 in true_group
            and i3 in true_group
        ):
            return True, group_id
        return False, None

    def _is_one_away(self, indices: Sequence[int]) -> bool:
        counts = np.zeros(4, dtype=np.int32)
        for idx in indices:
            group_id = int(self._group_by_word[idx])
            if group_id < 0 or group_id >= 4:
                continue
            if self._group_solved_mask[group_id] == 1:
                continue
            counts[group_id] += 1
        return bool((counts == 3).any())

    def _resolve_action(
        self,
        action: Union[int, np.integer, Sequence[int]],
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
        if isinstance(action, (int, np.integer)):
            action_idx = int(action)
            if action_idx < 0 or action_idx >= len(self._all_actions):
                return None, "action_index_out_of_range"
            selected = tuple(int(v) for v in self._all_actions[action_idx])
        else:
            if isinstance(action, (str, bytes)):
                return None, "unsupported_action_type"
            try:
                iterator = iter(action)
                a = int(next(iterator))
                b = int(next(iterator))
                c = int(next(iterator))
                d = int(next(iterator))
                try:
                    next(iterator)
                    return None, "invalid_action_arity"
                except StopIteration:
                    pass
            except Exception:
                return None, "unsupported_action_type"
            if a < 0 or a >= 16 or b < 0 or b >= 16 or c < 0 or c >= 16 or d < 0 or d >= 16:
                return None, "word_index_out_of_range"
            if a == b or a == c or a == d or b == c or b == d or c == d:
                return None, "duplicate_word_indices"

            # Sort network for 4 elements to avoid temporary list allocations.
            if a > b:
                a, b = b, a
            if c > d:
                c, d = d, c
            if a > c:
                a, c = c, a
            if b > d:
                b, d = d, b
            if b > c:
                b, c = c, b

            selected = (a, b, c, d)

        if any(self._word_solved_mask[idx] == 1 for idx in selected):
            return selected, "contains_solved_word"

        return selected, None

    def _effective_mask_mode(self) -> ActionMaskMode:
        mode = self._action_mask_mode
        if mode == ActionMaskMode.AUTO:
            if self._mask_strictness == MaskStrictness.STRICT:
                return ActionMaskMode.CONSISTENT
            if self._mask_strictness == MaskStrictness.OPEN:
                return ActionMaskMode.ALL
            return ActionMaskMode.VALID
        return mode

    @staticmethod
    def _parse_mask_mode(mode: Union[str, ActionMaskMode]) -> ActionMaskMode:
        if isinstance(mode, ActionMaskMode):
            return mode
        if not isinstance(mode, str):
            raise ValueError("action_mask_mode must be one of: auto, all, valid, consistent.")
        clean = mode.strip().lower()
        if clean == "all":
            return ActionMaskMode.ALL
        if clean == "valid":
            return ActionMaskMode.VALID
        if clean == "consistent":
            return ActionMaskMode.CONSISTENT
        if clean == "auto":
            return ActionMaskMode.AUTO
        raise ValueError("action_mask_mode must be one of: auto, all, valid, consistent.")

    @staticmethod
    def _parse_mask_strictness(mode: Union[str, MaskStrictness]) -> MaskStrictness:
        if isinstance(mode, MaskStrictness):
            return mode
        if not isinstance(mode, str):
            raise ValueError("mask_strictness must be one of: strict, balanced, open.")
        clean = mode.strip().lower()
        if clean == "strict":
            return MaskStrictness.STRICT
        if clean == "balanced":
            return MaskStrictness.BALANCED
        if clean == "open":
            return MaskStrictness.OPEN
        raise ValueError("mask_strictness must be one of: strict, balanced, open.")

    @staticmethod
    def _parse_one_away_mode(mode: Union[str, OneAwayMode]) -> OneAwayMode:
        if isinstance(mode, OneAwayMode):
            return mode
        if not isinstance(mode, str):
            raise ValueError(
                "one_away_mode must be one of: disabled, info_only, observation, reward_shaping."
            )
        clean = mode.strip().lower()
        if clean == "disabled":
            return OneAwayMode.DISABLED
        if clean == "info_only":
            return OneAwayMode.INFO_ONLY
        if clean == "observation":
            return OneAwayMode.OBSERVATION
        if clean == "reward_shaping":
            return OneAwayMode.REWARD_SHAPING
        raise ValueError(
            "one_away_mode must be one of: disabled, info_only, observation, reward_shaping."
        )

    @staticmethod
    def _parse_category_label_visibility(
        mode: Union[str, CategoryLabelVisibility],
    ) -> CategoryLabelVisibility:
        if isinstance(mode, CategoryLabelVisibility):
            return mode
        if not isinstance(mode, str):
            raise ValueError("category_label_visibility must be one of: never, solved_only, always.")
        clean = mode.strip().lower()
        if clean == "never":
            return CategoryLabelVisibility.NEVER
        if clean == "solved_only":
            return CategoryLabelVisibility.SOLVED_ONLY
        if clean == "always":
            return CategoryLabelVisibility.ALWAYS
        raise ValueError("category_label_visibility must be one of: never, solved_only, always.")

    def _rand_index(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be > 0")
        if _HAS_GYMNASIUM and hasattr(self, "np_random"):
            return int(self.np_random.integers(upper))  # type: ignore[attr-defined]
        return int(self._rng.randrange(upper))

    def _compute_reward(
        self,
        *,
        is_correct: bool,
        one_away: bool,
        solved_puzzle: bool,
        mistakes_exhausted: bool,
    ) -> float:
        cfg = self.reward_config

        if cfg.reward_mode == RewardMode.SPARSE:
            if solved_puzzle:
                return float(cfg.win_reward)
            if mistakes_exhausted:
                return float(cfg.lose_reward)
            return 0.0

        reward = cfg.step_penalty
        if is_correct:
            reward += cfg.solved_group_reward
        else:
            if self._one_away_mode == OneAwayMode.REWARD_SHAPING and one_away:
                reward += cfg.one_away_bonus
            else:
                reward += cfg.wrong_group_penalty

        if solved_puzzle:
            reward += cfg.win_reward
        elif mistakes_exhausted:
            reward += cfg.lose_reward
        return float(reward)

    def _encode_word(self, word: str) -> np.ndarray:
        out = np.full(self.max_word_length, fill_value=self.WORD_PAD_TOKEN, dtype=np.uint8)
        truncated = word[: self.max_word_length]
        for idx, ch in enumerate(truncated):
            if not ("a" <= ch <= "z"):
                raise ValueError(f"Unsupported character {ch!r} in word {word!r}")
            out[idx] = ord(ch) - ord("a")
        return out

    def _encode_label(self, label: str) -> np.ndarray:
        out = np.full(self.max_label_length, fill_value=self.LABEL_PAD_TOKEN, dtype=np.uint8)
        clean = label.strip().lower()
        pos = 0
        for ch in clean:
            if pos >= self.max_label_length:
                break
            if "a" <= ch <= "z":
                out[pos] = ord(ch) - ord("a")
                pos += 1
            elif ch in (" ", "-", "_"):
                continue
            else:
                continue
        return out

    def _refresh_visible_labels(self) -> None:
        if self._category_label_visibility == CategoryLabelVisibility.ALWAYS:
            self._labels_encoded[:] = self._true_labels_encoded
            return

        if self._category_label_visibility == CategoryLabelVisibility.NEVER:
            self._labels_encoded.fill(self.LABEL_UNKNOWN_TOKEN)
            return

        self._labels_encoded.fill(self.LABEL_UNKNOWN_TOKEN)
        for group_idx in range(4):
            if self._group_solved_mask[group_idx] == 1:
                self._labels_encoded[group_idx] = self._true_labels_encoded[group_idx]
