#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import numpy as np

from connections_rl import ConnectionsEnv
from connections_rl.puzzles import PuzzleBank, make_puzzle_bank


@dataclass
class TrainConfig:
    episodes: int = 5000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 4000
    eval_every: int = 500
    eval_episodes: int = 100
    seed: int = 0


def _normalize_key(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _is_ascii_word(word: str) -> bool:
    return all("a" <= ch <= "z" for ch in word)


def load_kaggle_connections_csv(csv_path: str) -> PuzzleBank:
    """
    Convert Kaggle Connections_Data.csv rows into `make_puzzle_bank(...)` schema.

    Expected columns (case/spacing-insensitive):
      - Game ID
      - Word
      - Group Name
      - Group Level
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        key_map: Dict[str, str] = {_normalize_key(k): k for k in reader.fieldnames}
        required = {
            "gameid": "Game ID",
            "word": "Word",
            "groupname": "Group Name",
            "grouplevel": "Group Level",
        }
        for norm, display in required.items():
            if norm not in key_map:
                raise ValueError(f"Missing required column: {display}")

        game_col = key_map["gameid"]
        word_col = key_map["word"]
        group_name_col = key_map["groupname"]
        group_level_col = key_map["grouplevel"]

        grouped: DefaultDict[str, List[Tuple[int, str, str]]] = defaultdict(list)
        for row in reader:
            game_id = str(row.get(game_col, "")).strip()
            raw_word = str(row.get(word_col, "")).strip().lower()
            group_name = str(row.get(group_name_col, "")).strip().upper()
            raw_level = str(row.get(group_level_col, "")).strip()

            if not game_id or not raw_word or not group_name or not raw_level:
                continue
            try:
                level = int(raw_level)
            except ValueError:
                continue

            # Current env schema accepts alphabetic words only.
            if not raw_word.isalpha() or not _is_ascii_word(raw_word):
                continue

            grouped[game_id].append((level, group_name, raw_word))

    puzzles: List[Dict[str, object]] = []
    for game_id, rows in grouped.items():
        by_level: DefaultDict[int, List[Tuple[str, str]]] = defaultdict(list)
        for level, group_name, word in rows:
            by_level[level].append((group_name, word))

        if sorted(by_level.keys()) != [0, 1, 2, 3]:
            continue

        groups: List[Dict[str, object]] = []
        valid = True
        for level in [0, 1, 2, 3]:
            entries = by_level[level]
            if len(entries) != 4:
                valid = False
                break

            labels = {name for name, _ in entries}
            if len(labels) != 1:
                valid = False
                break

            words = [word for _, word in entries]
            if len(set(words)) != 4:
                valid = False
                break

            groups.append({"label": next(iter(labels)), "words": words})

        if not valid:
            continue

        flat = [w for g in groups for w in g["words"]]
        if len(flat) != 16 or len(set(flat)) != 16:
            continue

        puzzles.append({"id": f"kaggle-{game_id}", "groups": groups})

    if not puzzles:
        raise ValueError("No valid puzzles could be parsed from CSV.")

    return make_puzzle_bank(puzzles)


def _state_from_obs(obs: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
    # Compact tabular state from README-documented fields.
    solved_mask = obs["word_solved_mask"]
    word_bits = 0
    for i in range(16):
        if int(solved_mask[i]) == 1:
            word_bits |= 1 << i

    mistakes = int(obs["mistakes_used"][0])
    invalid = int(obs["invalid_actions_used"][0])
    return (word_bits, mistakes, invalid)


def _epsilon(episode: int, cfg: TrainConfig) -> float:
    if episode >= cfg.epsilon_decay_episodes:
        return cfg.epsilon_end
    frac = episode / max(1, cfg.epsilon_decay_episodes)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def _select_action(
    q_table: Dict[Tuple[int, int, int], np.ndarray],
    state: Tuple[int, int, int],
    action_mask: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    valid = np.flatnonzero(action_mask)
    if valid.size == 0:
        return int(rng.integers(action_mask.shape[0]))

    if float(rng.random()) < epsilon:
        return int(valid[rng.integers(valid.size)])

    q = q_table.setdefault(state, np.zeros(action_mask.shape[0], dtype=np.float32))
    best_valid_q = q[valid]
    max_q = best_valid_q.max()
    ties = valid[np.flatnonzero(best_valid_q == max_q)]
    return int(ties[rng.integers(ties.size)])


def evaluate_policy(env: ConnectionsEnv, q_table: Dict[Tuple[int, int, int], np.ndarray], episodes: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    returns: List[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            state = _state_from_obs(obs)
            mask = obs["action_mask"]
            action = _select_action(q_table, state, mask, epsilon=0.0, rng=rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        returns.append(total)

    return float(np.mean(returns))


def train_q_learning(env: ConnectionsEnv, cfg: TrainConfig) -> Dict[Tuple[int, int, int], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    q_table: Dict[Tuple[int, int, int], np.ndarray] = {}

    for episode in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=cfg.seed + episode)
        done = False
        eps = _epsilon(episode, cfg)

        while not done:
            state = _state_from_obs(obs)
            action = _select_action(q_table, state, obs["action_mask"], eps, rng)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            next_state = _state_from_obs(next_obs)

            q = q_table.setdefault(state, np.zeros(env.action_space.n, dtype=np.float32))
            next_q = q_table.setdefault(next_state, np.zeros(env.action_space.n, dtype=np.float32))

            if done:
                target = float(reward)
            else:
                next_valid = np.flatnonzero(next_obs["action_mask"])
                max_next = 0.0 if next_valid.size == 0 else float(next_q[next_valid].max())
                target = float(reward) + cfg.gamma * max_next

            q[action] += cfg.alpha * (target - q[action])
            obs = next_obs

        if cfg.eval_every > 0 and episode % cfg.eval_every == 0:
            score = evaluate_policy(env, q_table, cfg.eval_episodes, seed=cfg.seed + 100_000 + episode)
            print(f"episode={episode} epsilon={eps:.3f} eval_avg_return={score:.4f}")

    return q_table


def build_env(csv_path: str | None) -> ConnectionsEnv:
    if csv_path:
        bank = load_kaggle_connections_csv(csv_path)
        return ConnectionsEnv(
            puzzle_bank=bank,
            include_action_mask=True,
            action_mask_mode="valid",
            max_invalid_actions=16,
        )

    return ConnectionsEnv(
        include_action_mask=True,
        action_mask_mode="valid",
        max_invalid_actions=16,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular Q-learning baseline for ConnectionsEnv.")
    parser.add_argument("--kaggle-csv", type=str, default="", help="Path to Connections_Data.csv (optional).")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = TrainConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )

    env = build_env(args.kaggle_csv or None)
    q_table = train_q_learning(env, cfg)
    final_score = evaluate_policy(env, q_table, episodes=cfg.eval_episodes, seed=cfg.seed + 999_999)
    print(f"final_eval_avg_return={final_score:.4f} states={len(q_table)}")


if __name__ == "__main__":
    main()
