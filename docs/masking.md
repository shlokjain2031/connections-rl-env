# Connections RL masking semantics

## Modes

- `all`: every 4-subset action index is mask-valid.
- `valid`: only actions over currently unsolved words are valid.
- `consistent`: currently aliases `valid` (explicitly; logical-pruning reserved for future work).
- `auto`: selected by `mask_strictness`.

## Auto strictness mapping

- `strict` -> `consistent`
- `balanced` -> `valid`
- `open` -> `all`

## Design note

`consistent` is intentionally documented as an alias today. This avoids false precision and keeps future upgrade paths explicit when full logical pruning is implemented.
