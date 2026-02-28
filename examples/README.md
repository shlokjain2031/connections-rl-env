# Examples

## Tabular Q-learning baseline

This folder includes a minimal RL baseline using `ConnectionsEnv` and the APIs documented in the project README:

- `ConnectionsEnv(...)`
- `env.reset(seed=...)`
- `env.step(action)`
- `env.sample_valid_action()`
- `action_mask` from observations
- `make_puzzle_bank(...)` for custom puzzle data

Run with bundled sample puzzles:

```bash
python3 examples/train_q_learning_baseline.py --episodes 3000 --eval-every 300
```

## Using the Kaggle NYT dataset

Dataset page:

- https://www.kaggle.com/datasets/eric27n/the-new-york-times-connections

The script expects the Kaggle CSV file (`Connections_Data.csv`) and converts it into the env puzzle format.

Example flow (requires Kaggle CLI configured locally):

```bash
kaggle datasets download -d eric27n/the-new-york-times-connections -p data --unzip
python3 examples/train_q_learning_baseline.py --kaggle-csv data/Connections_Data.csv --episodes 5000
```

Notes:

- The converter uses dataset fields described on Kaggle (`Game ID`, `Word`, `Group Name`, `Group Level`).
- The env currently accepts alphabetic words only. Rows with non-alphabetic words are skipped.
- A puzzle is used only if it forms exactly 4 groups of 4 unique words (16 unique words total).
