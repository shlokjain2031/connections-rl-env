# Connections RL benchmarks and guardrails

## Throughput benchmark

Run:

```bash
python3 benchmarks/benchmark_connections_env.py
```

Default benchmark sizes are CI-oriented:

- single env per mode: 250,000 steps
- vector env: 60,000 iterations
- vector env count: 16

## Guardrail (with warmup)

Run measurement only:

```bash
python3 scripts/perf_guardrail_connections.py
```

Run with baseline comparison:

```bash
python3 scripts/perf_guardrail_connections.py --baseline-json benchmarks/connections_perf_baseline.json --min-ratio 0.8
```

Guardrail behavior:

- warms up before measurement
- checks each metric independently (`single_all`, `single_valid`, `single_consistent`, `single_auto`, `vector_auto`)
- fails when measured value is below `baseline * min_ratio`

## Baseline generation

Generate/update baseline metrics on target CI hardware:

```bash
python3 scripts/gen_connections_baseline.py --output benchmarks/connections_perf_baseline.json
```

Commit that baseline with the CI config for stable regression detection.
