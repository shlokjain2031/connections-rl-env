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


def _single_sps(steps: int, warmup_steps: int, mode: ActionMaskMode) -> float:
    env = ConnectionsEnv(action_mask_mode=mode, include_action_mask=True, include_info=False)
    env.reset(seed=0)

    for _ in range(warmup_steps):
        action = env.sample_valid_action()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=0)

    start = time.perf_counter()
    for _ in range(steps):
        action = env.sample_valid_action()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=0)
    elapsed = time.perf_counter() - start
    print("mask cache size:", len(env._mask_cache), file=sys.stderr)
    return steps / elapsed


def _vector_sps(steps: int, warmup_steps: int, num_envs: int) -> float:
    vec = ConnectionsVectorEnv(num_envs=num_envs, include_action_mask=True, include_info=False)
    vec.reset(seed=0)

    for _ in range(warmup_steps):
        actions = [env.sample_valid_action() for env in vec.envs]
        vec.step(actions)

    total_steps = steps * num_envs
    start = time.perf_counter()
    for _ in range(steps):
        actions = [env.sample_valid_action() for env in vec.envs]
        vec.step(actions)
    elapsed = time.perf_counter() - start
    if vec.envs:
        print("mask cache size:", len(vec.envs[0]._mask_cache), file=sys.stderr)
    vec.close()
    return total_steps / elapsed


def _measure(steps: int, warmup_steps: int, vector_steps: int, vector_envs: int) -> Dict[str, float]:
    return {
        "single_all": _single_sps(steps, warmup_steps, ActionMaskMode.ALL),
        "single_valid": _single_sps(steps, warmup_steps, ActionMaskMode.VALID),
        "single_consistent": _single_sps(steps, warmup_steps, ActionMaskMode.CONSISTENT),
        "single_auto": _single_sps(steps, warmup_steps, ActionMaskMode.AUTO),
        "vector_auto": _vector_sps(vector_steps, warmup_steps, vector_envs),
    }


def _load_json(path: str) -> Dict[str, float]:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Baseline JSON must be an object mapping metric names to numbers.")
    out: Dict[str, float] = {}
    for key, value in data.items():
        out[key] = float(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Performance guardrail for Connections RL env.")
    parser.add_argument("--steps", type=int, default=120000, help="Measured single-env steps per mode.")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Warmup steps before measurement.")
    parser.add_argument("--vector-steps", type=int, default=30000, help="Measured vector-env iterations.")
    parser.add_argument("--vector-envs", type=int, default=16, help="Number of vector envs.")
    parser.add_argument(
        "--baseline-json",
        type=str,
        default="",
        help="Path to baseline JSON metrics. If omitted, script prints measured metrics and exits 0.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.8,
        help="Fail if measured metric drops below baseline * min_ratio.",
    )
    args = parser.parse_args()

    measured = _measure(args.steps, args.warmup_steps, args.vector_steps, args.vector_envs)
    print(json.dumps({"measured": measured}, indent=2, sort_keys=True))

    if not args.baseline_json:
        print("No baseline provided. Guardrail check skipped.", file=sys.stderr)
        return 0

    baseline = _load_json(args.baseline_json)
    failures = []
    for key, value in measured.items():
        if key not in baseline:
            failures.append(f"Missing baseline metric: {key}")
            continue
        threshold = baseline[key] * args.min_ratio
        if value < threshold:
            failures.append(
                f"{key}: measured {value:.2f} < threshold {threshold:.2f} (baseline {baseline[key]:.2f})"
            )

    if failures:
        for message in failures:
            print(f"FAIL: {message}", file=sys.stderr)
        return 1

    print("PASS: all metrics above baseline ratio threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
