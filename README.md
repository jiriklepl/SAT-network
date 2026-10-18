# Synthesising Logic Programs with z3

This repository synthesizes and optimizes straight-line (SSA-style) boolean logic programs using Z3 (`sat_synth_cpp`).

> **Note:** The C++ implementation (`sat_synth_cpp`, documented in [`cpp/README.md`](cpp/README.md)) supersedes `main.py`, which is now deprecated.

## Datasets & CLI

Built-in declarative datasets are located in `cpp/datasets/*.json` and can be selected via `--dataset`:

- Built-ins: `adder`, `gol`, `gol1`, `gol2`, `sloppy-adder`, `sloppy-adder3`, `life`, `life-compressed`, `maze`, `maze-compressed`, `brian`, `brian-compressed`, `fire`, `fire-compressed`, `wire`, `wire-compressed`, `excitable`, `excitable-compressed`, `cyclic`, `cyclic-compressed`, `fluid`, `critters`, and `traffic`.
- Custom: pass a JSON path via `--config` (see [`cpp/README.md`](cpp/README.md) for the declarative quantified JSON schema).

Schema options:
- Built-in / parameterized generator:
  - GoL: `{ "type": "gol", "instructions": 14, "gol": { "left_range": 4, "center_range": 3, "right_range": 4, "include_alive": true } }`
  - Adder: `{ "type": "adder", "num_inputs": 4, "instructions": 6 }`
- Explicit examples: `{ "num_inputs": N, "num_outputs": M, "instructions": K, "examples": [{"inputs": [..], "outputs": [..]}, ...] }`
- Quantified specification: `{ "forall": [...], "let_row": [...], "where": "...", "inputs": ..., "outputs": ... }`

## Building & Running (`sat_synth_cpp`)

```bash
# Build the C++ release binary
cmake --preset release
cmake --build --preset release -j

# Solve built-in adder dataset with CEGIS
./build/release/sat_synth_cpp --dataset adder --cegis

# Solve incrementally with batches of 16 examples
./build/release/sat_synth_cpp --dataset gol --batch-size 16

# Use a custom config and override instruction budget
./build/release/sat_synth_cpp --config path/to/dataset.json --instructions 12

# Run native Catch2 unit & integration tests
ctest --test-dir build/release --output-on-failure
```

## Building Reference Assumptions (`build_assumptions.py`)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 build_assumptions.py --dataset life-compressed --output /tmp/life-compressed.assume
./build/release/sat_synth_cpp --dataset life-compressed --assume /tmp/life-compressed.assume --post-process
```

`build_assumptions.py` emits a complete program in the `--assume` file format accepted by `sat_synth_cpp` (and legacy `main.py`).
The default `circuit` strategy uses generated gate networks for supported cellular automata datasets and falls back to exact DNF for other datasets. Use `--strategy dnf` to force the truth-table construction. If the header reports `required-instructions`, pass that value with `--instructions` when it differs from the dataset default.
