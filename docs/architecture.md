# Connections RL architecture

`connections_rl` is a standalone RL environment package:

- Core env: `ConnectionsEnv`
- Puzzle schema/loader: `connections_rl/puzzles.py`
- Wrappers: `connections_rl/wrappers.py`
- Vector collector env: `connections_rl/vector_env.py`
- Registration: `ConnectionsRL-v0`

## Stage-2/3 task knobs

- Action mask mode: `all`, `valid`, `consistent`, `auto`
- Mask strictness (for `auto`): `strict`, `balanced`, `open`
- One-away exposure: `disabled`, `info_only`, `observation`, `reward_shaping`
- Category-label visibility: `never`, `solved_only`, `always`
- Reward mode: `sparse` (terminal-only), `shaped`

## Transition reason taxonomy

- `reset`
- `solved_group`
- `wrong_group`
- `invalid_action`
- `puzzle_solved`
- `mistakes_exhausted`
- `invalid_truncation`
- `max_steps_truncation`

## Reproducibility invariants

- `reset(seed=...)` controls deterministic puzzle sampling.
- `reset(options={"puzzle_id": ...})` overrides random sampling.
- Every reset/step info payload includes:
  - `puzzle_id`
  - `dataset_hash` (short)
  - `dataset_hash_full` (full SHA-256)
