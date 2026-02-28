#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from connections_rl import ActionMaskMode, ConnectionsEnv, ConnectionsVectorEnv


def run_single_mode_benchmark(steps: int, *, mode: ActionMaskMode) -> Dict[str, float]:
    env = ConnectionsEnv(action_mask_mode=mode, include_action_mask=True)
    env.reset(seed=0)

    start = time.perf_counter()
    for _ in range(steps):
        action = env.sample_valid_action()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=0)
    elapsed = time.perf_counter() - start

    return {
        "steps": float(steps),
        "seconds": elapsed,
        "steps_per_sec": steps / elapsed,
    }


def run_vector_benchmark(steps: int, num_envs: int) -> Dict[str, float]:
    vec = ConnectionsVectorEnv(num_envs=num_envs, include_action_mask=True)
    vec.reset(seed=0)
    total_steps = steps * num_envs

    start = time.perf_counter()
    for _ in range(steps):
        actions = [env.sample_valid_action() for env in vec.envs]
        vec.step(actions)
    elapsed = time.perf_counter() - start
    vec.close()

    return {
        "steps": float(total_steps),
        "seconds": elapsed,
        "steps_per_sec": total_steps / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Connections RL environment benchmarks.")
    parser.add_argument(
        "--single-steps",
        type=int,
        default=250000,
        help="Single-env steps per mask mode benchmark.",
    )
    parser.add_argument(
        "--vector-steps",
        type=int,
        default=60000,
        help="Vector-env iterations.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of envs for vector benchmark.",
    )
    args = parser.parse_args()

    results = {
        "single_all": run_single_mode_benchmark(args.single_steps, mode=ActionMaskMode.ALL),
        "single_valid": run_single_mode_benchmark(args.single_steps, mode=ActionMaskMode.VALID),
        "single_consistent": run_single_mode_benchmark(args.single_steps, mode=ActionMaskMode.CONSISTENT),
        "single_auto": run_single_mode_benchmark(args.single_steps, mode=ActionMaskMode.AUTO),
        "vector_auto": run_vector_benchmark(args.vector_steps, args.num_envs),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
