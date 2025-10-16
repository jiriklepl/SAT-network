# Constructing a "neural network" to compute Game of Life

This implementation uses the z3 SMT solver via the `z3-solver` Python package (replacing the previous PySAT-based version).

## Configuring a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Running the code

```bash
python3 main.py
```
