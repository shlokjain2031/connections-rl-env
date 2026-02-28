# Connections RL Environment

Connections environment for reinforcement learning with deterministic puzzle sampling, configurable masking behavior, and training-ready observations.

## What is included

- Canonical Connections-style 4x4 grouping gameplay (`16` words, `4` groups of `4`).
- Configurable action masks:
	- `all`: every 4-subset action is mask-valid.
	- `valid`: only actions over currently unsolved words are mask-valid.
	- `consistent`: currently aliases `valid` (reserved for stricter logical pruning in future).
	- `auto`: mode selected from `mask_strictness`.
- Strictness control for `auto` mode:
	- `strict` -> `consistent`
	- `balanced` -> `valid`
	- `open` -> `all`
- One-away controls:
	- `disabled`, `info_only`, `observation`, `reward_shaping`
- Category-label visibility controls:
	- `never`, `solved_only`, `always`
- Reward modes:
	- `sparse` (terminal-only)
	- `shaped` (step-level shaping)
- Finite invalid-action budget (`max_invalid_actions`) and step truncation (`max_steps`) to avoid non-terminating episodes.
- Deterministic puzzle sampling with `reset(seed=...)` and explicit override with `reset(options={"puzzle_id": ...})`.
- Dataset integrity metadata in `info` payloads:
	- `puzzle_id`
	- `dataset_hash`
	- `dataset_hash_full`
- Lightweight synchronous batched API via `ConnectionsVectorEnv`.
- Gymnasium registration and wrappers (when Gymnasium is installed).

## Installation

```bash
pip install -e .
```

Optional Gymnasium support:

```bash
pip install -e ".[gym]"
```

## Quick Start

```python
from connections_rl import ActionMaskMode, ConnectionsEnv

env = ConnectionsEnv(
		action_mask_mode=ActionMaskMode.AUTO,
		include_action_mask=True,
		max_invalid_actions=16,
)

obs, info = env.reset(seed=7)
done = False
while not done:
		action = env.sample_valid_action()
		obs, reward, terminated, truncated, info = env.step(action)
		done = terminated or truncated
```

## Batched collection

```python
from connections_rl import ConnectionsVectorEnv

vec = ConnectionsVectorEnv(num_envs=8, auto_reset=True, include_action_mask=True)
obs, infos = vec.reset(seed=123)

actions = [env.sample_valid_action() for env in vec.envs]
obs, rewards, terminated, truncated, infos = vec.step(actions)
```

## Gymnasium registration and wrappers

`ConnectionsRL-v0` is auto-registered when `connections_rl` is imported and Gymnasium is installed.

```python
import gymnasium as gym
import connections_rl
from connections_rl import ActionMaskToInfoWrapper, FlattenConnectionsObservation

env = gym.make("ConnectionsRL-v0")
env = ActionMaskToInfoWrapper(env)
flat_env = FlattenConnectionsObservation(env)
```

## Observation schema

- `words`: `(16, max_word_length)` with `a=0..z=25`, pad=`26`
- `active_word_mask`: `(16,)` (`1` for unsolved)
- `word_solved_mask`: `(16,)`
- `group_solved_mask`: `(4,)`
- `group_labels`: `(4, max_label_length)` with unknown token=`27`
- `attempts_used`: `(1,)`
- `mistakes_used`: `(1,)`
- `invalid_actions_used`: `(1,)`
- `groups_solved_count`: `(1,)`
- `words_solved_count`: `(1,)`
- `one_away`: `(1,)` (optional; when `one_away_mode="observation"`)
- `action_mask`: `(num_actions,)` (optional)

## Core APIs

- `env.valid_action_mask()`
- `env.sample_valid_action()`
- `env.indices_to_action(indices)` / `env.action_to_indices(action_index)`
- `get_preset_env_kwargs(name)`
- `load_default_puzzle_bank()`, `make_puzzle_bank(...)`

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Benchmarks

```bash
python3 benchmarks/benchmark_connections_env.py
python3 scripts/perf_guardrail_connections.py
python3 scripts/gen_connections_baseline.py --output benchmarks/connections_perf_baseline.json
```
