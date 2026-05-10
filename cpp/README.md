# C++ Core Solver

`sat_synth_cpp` is a first-mile C++ port of the core solver path in `main.py`.
It supports JSON configs with explicit `examples` and C++-generated built-in
datasets selected by config `type` or `--dataset`.

It intentionally does not support Python plugin loading or quantified synthesis.
The `--post-process` flag currently runs mask-only simplification plus local SAT
resynthesis for small one-fanout windows.

## Build

```bash
cmake --preset debug
cmake --build --preset debug -j
```

The supported baseline is C++20, CMake 3.20 or newer, GCC 13 or newer or
Clang 17 or newer, and system Z3 headers/library. CMake finds Z3 and fetches
`nlohmann/json`, `cxxopts`, and Catch2 with `FetchContent`.
Boost.DynamicBitset is header-only.

## Dataset Support

C++ built-in dataset generation is intended to match Python plugin output
exactly for deterministic configs, including row order and don't-care outputs.
Full state-space generation is supported when `max_examples` is absent or is at
least the full state-space size. Sampled state-space configs with
`max_examples` smaller than the full state space are intentionally unsupported
for now because the C++ generator does not reproduce Python's RNG sampling.

## Layout

- `sat_synth_cpp.cpp`: executable orchestration, CEGIS loop, and batch solving loop.
- `cli.*`: `cxxopts` command-line parsing and user-facing logging.
- `config.*`: JSON config loading and dataset dumping.
- `datasets.*`: C++ built-in dataset generators.
- `program.*`: program/data model, operator table, text emission, and packed verifier.
- `postprocess.*`: post-processing beam orchestration and public options.
- `postprocess_common.*`, `postprocess_mask.*`, `postprocess_resynthesis.*`:
  scoring, shared candidate utilities, mask-only simplification, and local SAT
  resynthesis.
- `encoding.*`: Z3 structure/example encoding, solver construction, and model extraction.
- `solver.*`: solver orchestration, CEGIS, batching, timing, and verification result packaging.
- `tests/`: modular Catch2 unit tests for core modules.

## Run

```bash
build/sat_synth_cpp --config path/to/config.json --cegis --cegis-initial-size 1
build/sat_synth_cpp --dataset adder --cegis
build/sat_synth_cpp --config typed-dataset.json --dump-dataset
build/sat_synth_cpp --config path/to/config.json --assume path/to/program.txt
build/sat_synth_cpp --config path/to/config.json --make-smt2
build/sat_synth_cpp --config path/to/config.json --make-dimacs
build/sat_synth_cpp --config path/to/config.json --make-blif
build/sat_synth_cpp --config path/to/config.json --output-blif
build/sat_synth_cpp --config path/to/config.json --post-process --post-process-resynthesis-maxnodes 5
build/sat_synth_cpp --config path/to/config.json --post-process --post-process-score 'program-length;max-output-depth,operator-cost'
build/sat_synth_cpp --config path/to/config.json --post-process --post-process-replace-patience 50
build/sat_synth_cpp --config path/to/config.json --profile
build/sat_synth_cpp --list-datasets
```

The emitted program uses the same text format as `main.py`:

```text
T0: XOR(I0, I1)
OUT0: T0
```

Assumption files use the same text format. Blank lines and lines starting with
`#` are ignored. Use `--assume -` to read assumptions from stdin.

`--profile` prints phase timings and counters to stderr. It covers dataset
generation, structure encoding, example packing/encoding, Z3 solve time, model
extraction, post-processing, per-generator post-processing counters, packed
verification, and Z3 bit-vector literal cache hits/misses.

`--post-process` runs deterministic mask-only simplifications after model
extraction and before text or BLIF emission. It can remove unreachable
instructions, redirect equivalent masks to earlier sources, apply simple
`AND`/`OR`/`XOR` algebra, and simplify output selectors under don't-care masks.
It also tries structural regrouping/bypass generators, deterministic replacement
search, and local SAT resynthesis for closed one-fanout windows. Use
`--post-process-replace-patience`, `--post-process-resynthesis-maxnodes`,
`--post-process-resynthesis-patience`, and `--generator-timeout` to control
that search. `--generator-timeout` is applied independently to the cheap
mask/structural generator, replacement search, and local SAT resynthesis.

`--post-process-score` selects lexicographic score metrics. Separate metrics in
one phase with commas and phases with semicolons. Prefix a metric with `-` for
descending order. Supported metrics are `program-length`, `output-depth`,
`max-output-depth`, `sum-output-depth`, `total-node-depth`, `total-tree-size`,
`operator-cost`, `xor-count`, `output-cone-size`, `max-output-cone-size`,
`sum-output-cone-size`, `fanout`, `max-fanout`, `sum-fanout`,
`one-fanout-count`, `independent-pairs`, `entropy`, and `random`. The C++
`random` metric is deterministic for a fixed `--seed`.

## Test

```bash
ctest --test-dir build/debug --output-on-failure
```

CTest runs both the C++ unit test executable and the Python integration tests.
They can also be run directly:

```bash
build/debug/cpp_unit_tests
SAT_SYNTH_CPP=build/debug/sat_synth_cpp python3 -m unittest test_cpp_solver
```
