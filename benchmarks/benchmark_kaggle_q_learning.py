#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from dataclasses import asdict
from hashlib import sha256
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from connections_rl import ConnectionsEnv
from connections_rl.puzzles import Puzzle, PuzzleBank
from examples.train_q_learning_baseline import (
    TrainConfig,
    _epsilon,
    _select_action,
    _state_from_obs,
    load_kaggle_connections_csv,
)


def split_puzzle_bank(bank: PuzzleBank, train_ratio: float, seed: int) -> Tuple[PuzzleBank, PuzzleBank]:
    puzzles = list(bank.puzzles)
    if len(puzzles) < 2:
        raise ValueError("Need at least 2 puzzles to create a train/test split.")

    rng = np.random.default_rng(seed)
    order = np.arange(len(puzzles), dtype=np.int32)
    rng.shuffle(order)

    split = int(round(len(puzzles) * train_ratio))
    split = max(1, min(len(puzzles) - 1, split))

    train_puzzles = [puzzles[int(i)] for i in order[:split]]
    test_puzzles = [puzzles[int(i)] for i in order[split:]]

    def _mk(subset: Sequence[Puzzle], name: str) -> PuzzleBank:
        ids = "|".join(p.puzzle_id for p in subset)
        digest = sha256(f"{name}:{ids}".encode("utf-8")).hexdigest()
        return PuzzleBank(puzzles=tuple(subset), dataset_hash_full=digest)

    return _mk(train_puzzles, "train"), _mk(test_puzzles, "test")


def build_env(bank: PuzzleBank, action_mask_mode: str) -> ConnectionsEnv:
    return ConnectionsEnv(
        puzzle_bank=bank,
        include_action_mask=True,
        action_mask_mode=action_mask_mode,
        max_invalid_actions=16,
    )


def evaluate_detailed(
    env: ConnectionsEnv,
    q_table: Dict[Tuple[int, int, int], np.ndarray],
    episodes: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    solved = 0
    groups_total = 0.0
    steps_total = 0
    solved_steps: List[int] = []
    return_total = 0.0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_steps = 0

        while not done:
            state = _state_from_obs(obs)
            action = _select_action(q_table, state, obs["action_mask"], epsilon=0.0, rng=rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_steps += 1
            done = bool(terminated or truncated)

        groups = int(obs["groups_solved_count"][0])
        is_solved = groups == 4
        if is_solved:
            solved += 1
            solved_steps.append(ep_steps)

        groups_total += float(groups)
        steps_total += ep_steps
        return_total += ep_return

    solved_steps_mean = float(np.mean(solved_steps)) if solved_steps else float("nan")

    return {
        "episodes": float(episodes),
        "solve_rate": solved / float(episodes),
        "groups_solved_per_episode": groups_total / float(episodes),
        "avg_steps_per_episode": steps_total / float(episodes),
        "steps_to_solve": solved_steps_mean,
        "avg_return": return_total / float(episodes),
    }


def train_with_curve(
    env: ConnectionsEnv,
    eval_env: ConnectionsEnv,
    cfg: TrainConfig,
    eval_every: int,
    eval_episodes: int,
) -> Tuple[Dict[Tuple[int, int, int], np.ndarray], List[Dict[str, float]]]:
    rng = np.random.default_rng(cfg.seed)
    q_table: Dict[Tuple[int, int, int], np.ndarray] = {}
    curve: List[Dict[str, float]] = []

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

        if eval_every > 0 and episode % eval_every == 0:
            test_metrics = evaluate_detailed(
                eval_env,
                q_table,
                episodes=eval_episodes,
                seed=cfg.seed + 900_000 + episode,
            )
            point = {
                "episode": float(episode),
                "epsilon": float(eps),
                "test_solve_rate": float(test_metrics["solve_rate"]),
                "test_groups_solved_per_episode": float(test_metrics["groups_solved_per_episode"]),
                "test_avg_steps_per_episode": float(test_metrics["avg_steps_per_episode"]),
                "test_avg_return": float(test_metrics["avg_return"]),
            }
            curve.append(point)
            print(
                "episode="
                f"{episode} epsilon={eps:.3f} test_solve_rate={point['test_solve_rate']:.4f} "
                f"test_groups={point['test_groups_solved_per_episode']:.4f}"
            )

    return q_table, curve


def learning_curve_slope(curve: Sequence[Dict[str, float]]) -> float:
    if len(curve) < 2:
        return float("nan")
    x = np.array([row["episode"] for row in curve], dtype=np.float64)
    y = np.array([row["test_solve_rate"] for row in curve], dtype=np.float64)
    coeff = np.polyfit(x, y, deg=1)
    return float(coeff[0])


def write_curve_csv(path: pathlib.Path, curve: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "epsilon",
        "test_solve_rate",
        "test_groups_solved_per_episode",
        "test_avg_steps_per_episode",
        "test_avg_return",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in curve:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle benchmark for tabular Q-learning on ConnectionsEnv.")
    parser.add_argument("--kaggle-csv", type=str, default="data/Connections_Data.csv")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--final-eval-episodes", type=int, default=500)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-mask-mode", type=str, default="valid", choices=["all", "valid", "consistent", "auto"])
    parser.add_argument("--metrics-out", type=str, default="benchmarks/kaggle_q_learning_metrics.json")
    parser.add_argument("--curve-out", type=str, default="benchmarks/kaggle_q_learning_curve.csv")
    args = parser.parse_args()

    full_bank = load_kaggle_connections_csv(args.kaggle_csv)
    train_bank, test_bank = split_puzzle_bank(full_bank, train_ratio=args.train_ratio, seed=args.seed)

    train_env = build_env(train_bank, action_mask_mode=args.action_mask_mode)
    test_env = build_env(test_bank, action_mask_mode=args.action_mask_mode)

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

    q_table, curve = train_with_curve(
        env=train_env,
        eval_env=test_env,
        cfg=cfg,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )

    train_eval = evaluate_detailed(
        train_env,
        q_table,
        episodes=args.final_eval_episodes,
        seed=args.seed + 1_000_000,
    )
    test_eval = evaluate_detailed(
        test_env,
        q_table,
        episodes=args.final_eval_episodes,
        seed=args.seed + 2_000_000,
    )

    gap = float(train_eval["solve_rate"] - test_eval["solve_rate"])
    slope = learning_curve_slope(curve)

    results = {
        "config": asdict(cfg),
        "dataset": {
            "source_csv": args.kaggle_csv,
            "dataset_hash_full": full_bank.dataset_hash_full,
            "total_puzzles": len(full_bank.puzzles),
            "train_puzzles": len(train_bank.puzzles),
            "test_puzzles": len(test_bank.puzzles),
            "train_ratio": args.train_ratio,
            "split_seed": args.seed,
        },
        "environment": {
            "action_mask_mode": args.action_mask_mode,
            "include_action_mask": True,
            "max_invalid_actions": 16,
        },
        "metrics": {
            "train": train_eval,
            "test": test_eval,
            "generalization_gap": gap,
            "learning_curve_slope": slope,
            "num_q_states": len(q_table),
        },
        "curve_points": curve,
    }

    metrics_path = pathlib.Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    write_curve_csv(pathlib.Path(args.curve_out), curve)

    print(json.dumps({"metrics_out": str(metrics_path), "curve_out": args.curve_out}, indent=2))
    print(
        "final "
        f"train_solve_rate={train_eval['solve_rate']:.4f} "
        f"test_solve_rate={test_eval['solve_rate']:.4f} "
        f"gap={gap:.4f} slope={slope:.8f} "
        f"test_groups={test_eval['groups_solved_per_episode']:.4f} "
        f"test_steps_to_solve={test_eval['steps_to_solve']}"
    )


if __name__ == "__main__":
    main()
