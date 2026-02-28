# Connections RL Environment

Standalone project for the Connections RL environment.

This folder is intentionally isolated from the Wordle project code.

## Install

```bash
pip install -e .
```

Optional Gymnasium support:

```bash
pip install -e ".[gym]"
```

## Run tests

```bash
python3 -m unittest discover -s tests -v
```

## Run benchmarks

```bash
python3 benchmarks/benchmark_connections_env.py
python3 scripts/perf_guardrail_connections.py
python3 scripts/gen_connections_baseline.py --output benchmarks/connections_perf_baseline.json
```

## Docs

- [Architecture](docs/architecture.md)
- [Masking semantics](docs/masking.md)
- [Benchmarks and guardrails](docs/benchmarks.md)
