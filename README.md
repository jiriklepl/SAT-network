# Synthesising Logic Programs with z3

This implementation encodes straight-line (SSA-style) logic programs and solves them with z3 (`z3-solver`) using a custom tactic pipeline (`simplify → propagate-values → bit-blast → sat`).

## Datasets & CLI

Built-in datasets are registered in `dataset_plugins/` and can be selected via CLI:

- Built-ins: `adder`, `gol`, `gol1`, `gol2`, `sloppy-adder`, `sloppy-adder3`, `life`, `life-compressed`, `maze`, `maze-compressed`, `brian`, `brian-compressed`, `fire`, `fire-compressed`, `wire`, `wire-compressed`, `excitable`, `excitable-compressed`, `cyclic`, `cyclic-compressed`, `fluid`, `critters`, and `traffic`.
- Custom: pass a JSON path via `--config`.

Schema options:
- Generator-based (recommended for brevity):
  - GoL: `{ "type": "gol", "instructions": 14, "gol": { "left_range": 4, "center_range": 3, "right_range": 4, "include_alive": true } }`
  - Adder: `{ "type": "adder", "num_inputs": 3, "instructions": 5 }`
- Explicit examples: `{ "num_inputs": N, "num_outputs": M, "instructions": K, "examples": [{"inputs": [..], "outputs": [..]}, ...] }`

If `instructions` is omitted the program length falls back to the CLI default (currently 16). You can override it at runtime with `--instructions`.

## Configuring a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Running the code

```bash
# Use default GoL config
python3 main.py

# Choose built-in adder dataset
python3 main.py --dataset adder

# Solve incrementally with batches of 16 examples
python3 main.py --dataset gol --batch-size 16

# Use a custom config and override instruction budget
python3 main.py --config path/to/dataset.json --instructions 12

# Use the plain Z3 QF_BV solver
python3 main.py --solver z3
```
