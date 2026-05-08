# C++ Core Solver

`sat_synth_cpp` is a first-mile C++ port of the core solver path in `main.py`.
It supports JSON configs with explicit `examples` and C++-generated built-in
datasets selected by config `type` or `--dataset`.

It intentionally does not support Python plugin loading, quantified synthesis,
post-processing, assumption files, BLIF output, SMT-LIB export, or DIMACS export.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

CMake finds the system Z3 installation and fetches `nlohmann/json`, `cxxopts`,
and Catch2 with `FetchContent`. Boost.Multiprecision is header-only.

## Layout

- `sat_synth_cpp.cpp`: executable orchestration, CEGIS loop, and batch solving loop.
- `cli.*`: `cxxopts` command-line parsing and user-facing logging.
- `config.*`: JSON config loading and dataset dumping.
- `datasets.*`: C++ built-in dataset generators.
- `program.*`: program/data model, operator table, text emission, and packed verifier.
- `encoding.*`: Z3 structure/example encoding, solver construction, and model extraction.
- `solver.*`: solver orchestration, CEGIS, batching, timing, and verification result packaging.
- `tests/`: modular Catch2 unit tests for core modules.

## Run

```bash
build/sat_synth_cpp --config path/to/config.json --cegis --cegis-initial-size 1
build/sat_synth_cpp --dataset adder --cegis
build/sat_synth_cpp --config typed-dataset.json --dump-dataset
build/sat_synth_cpp --list-datasets
```

The emitted program uses the same text format as `main.py`:

```text
T0: XOR(I0, I1)
OUT0: T0
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

CTest runs both the C++ unit test executable and the Python integration tests.
They can also be run directly:

```bash
build/cpp_unit_tests
SAT_SYNTH_CPP=build/sat_synth_cpp python3 -m unittest test_cpp_solver
```
