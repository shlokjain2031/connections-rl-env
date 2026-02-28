#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Dict


def _extract_measured(payload: Dict[str, object]) -> Dict[str, float]:
    measured = payload.get("measured")
    if not isinstance(measured, dict):
        raise ValueError("Guardrail output missing measured metrics.")
    return {k: float(v) for k, v in measured.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate baseline JSON for Connections perf guardrail.")
    parser.add_argument("--output", type=str, default="benchmarks/connections_perf_baseline.json")
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--vector-steps", type=int, default=30000)
    parser.add_argument("--vector-envs", type=int, default=16)
    args = parser.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo / "scripts" / "perf_guardrail_connections.py"),
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--vector-steps",
        str(args.vector_steps),
        "--vector-envs",
        str(args.vector_envs),
    ]

    proc = subprocess.run(cmd, cwd=str(repo), text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    payload = json.loads(proc.stdout)
    measured = _extract_measured(payload)
    out_path = repo / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(measured, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote baseline metrics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
