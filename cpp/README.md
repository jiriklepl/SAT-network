# C++ Core Solver (`sat_synth_cpp`)

`sat_synth_cpp` is the primary C++ solver and post-processor for straight-line (SSA) boolean program synthesis, superseding the legacy `main.py` implementation.

It supports declarative JSON configs with explicit examples or first-order quantified specifications, as well as built-in datasets selected by config `type` or `--dataset`.

For quantified specifications with large or unmaterialized domains, it features a hybrid symbolic CEGIS engine (`SymbolicVerifier`) that couples quantifier-free bit-vector (`QF_BV`) miter checks with on-the-fly basis bitset accumulation, as well as two-tier counterexample-guided post-processing candidate verification.

## Dependencies & Setup

### Linux (Ubuntu / Debian)

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build g++-15 libz3-dev libboost-all-dev
```

(GCC 14+/15+ or Clang 18+/22+ with C++23/26 support is recommended.)

## Build

```bash
cmake --preset release
cmake --build --preset release -j
```

Available presets in `CMakePresets.json`:
- `debug`: `-O0 -g` debug build in `build/debug`
- `debug-sanitize`: AddressSanitizer + UndefinedBehaviorSanitizer build in `build/debug-sanitize`
- `release-with-debug-info`: `-O3 -g -DNDEBUG` build in `build/release-with-debug-info`
- `release`: `-O3 -DNDEBUG` production build in `build/release`

CMake finds system Z3 and Boost (`boost/dynamic_bitset.hpp`) and fetches `nlohmann/json`, `cxxopts`, and `Catch2` with `FetchContent`.

For local sanitizer validation with AddressSanitizer and UndefinedBehaviorSanitizer:

```bash
cmake --preset debug-sanitize
cmake --build --preset debug-sanitize -j2
ctest --test-dir build/debug-sanitize --output-on-failure
```

## Authoring Declarative JSON Datasets

Declarative JSON dataset configurations (in `cpp/datasets/*.json` or passed via `--config`) are the official replacement for legacy Python dataset plugins (`dataset_plugins/*.py`).

A dataset config can either provide explicit `examples` or a declarative quantified specification (`forall` + `inputs` + `outputs`):

```json
{
  "instructions": 32,
  "defaults": {
    "excitable": {
      "states": 8
    }
  },
  "params": {
    "state_count": { "path": "excitable.states", "default": 8 }
  },
  "constraints": [
    {
      "check": "state_count >= 3",
      "error": "excitable.states must be at least 3"
    }
  ],
  "let": {
    "bits_per_state": "required_bits(state_count)"
  },
  "num_inputs": "9 * bits_per_state",
  "num_outputs": "bits_per_state",
  "sample_section": "excitable",
  "forall": [
    { "var": "states", "tuple": { "size": 9, "range": [0, "state_count"] } }
  ],
  "let_row": {
    "center": "states[0]",
    "next_state": "center == 0 ? (count(states, 1, 1) > 0 ? 1 : 0) : (center == state_count - 1 ? 0 : center + 1)"
  },
  "inputs": [
    "state_bits(states, bits_per_state)"
  ],
  "outputs": [
    "bits(next_state, bits_per_state)"
  ]
}
```

### Schema Reference

- **`defaults`** *(object, optional)*: Default runtime configuration object returned by `default_dataset_config(name)`.
- **`params`** *(object, optional)*: Parameter bindings `{"<name>": {"path": "<dotted.path>" | ["<path1>", ...], "default": <int|bool>}}` resolved against the runtime config.
- **`constraints`** *(array, optional)*: Static validation rules `{"check": "<expr>", "error": "<message>"}` evaluated before building the dataset.
- **`let`** *(object, optional)*: Static scalar constants evaluated in dependency order before quantification.
- **`num_inputs`** / **`num_outputs`** *(integer or expression string)*: Declared circuit input and output bit widths.
- **`instructions`** *(integer or expression string, optional)*: Default SSA instruction budget (defaults to `16`).
- **`forall`** *(array of domains)*: Quantified variables spanning the input domain:
  - `{"var": "x", "range": ["<start_expr>", "<end_expr>"]}`: Scalar integer in `[start, end - 1]`.
  - `{"var": "x", "values": ["<expr1>", "<expr2>", ...]}`: Scalar integer taking each listed value in order.
  - `{"var": "x", "tuple": {"size": "<size_expr>", "range": ["<start_expr>", "<end_expr>"]}}`: Tuple of `size` integers each in `[start, end - 1]`.
- **`let_row`** *(object, optional)*: Per-row intermediate scalar bindings `{"<name>": "<expr>"}` evaluated in dependency order for each point.
- **`where`** *(expression string, optional)*: Boolean domain filter predicate. Only assignments where `where` evaluates to non-zero are included.
- **`inputs`** / **`outputs`** *(array of expression strings)*: Bit-emitting expressions defining the circuit inputs and expected outputs (`null` denotes a don't-care output bit).

### Supported Expressions & Built-in Functions

- **Variables & Indexing**: Reference scalar variables by identifier (`state_count`, `center`) and index tuple variables with `states[0]`. Literals include integers, `true` (`1`), `false` (`0`), and `null` (don't-care output).
- **Operators** (C-style precedence):
  - Ternary conditional: `cond ? then_expr : else_expr`
  - Logical: `||`, `&&`, `!`
  - Bitwise: `|`, `^`, `&`, `~`, `<<`, `>>`
  - Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
  - Arithmetic: `+`, `-`, `*`, `/` (truncating integer division), `%` (modulo)
- **Built-in Functions**:
  - `required_bits(n)`: Minimum bit width needed to represent states `0 .. n - 1` (`max(1, bit_width(n - 1))`).
  - `popcount(x)`: 64-bit population count of `x`.
  - `count(tuple_var, target[, start_index])`: Count elements in `tuple_var[start_index:]` equal to `target`.
  - `bit(expr)`: Emit a single boolean bit (`expr != 0`).
  - `bits(value, width)`: Emit `width` LSB-first bits `((value >> i) & 1) != 0` for `i in 0 .. width - 1`.
  - `state_bits(tuple_var, width)`: For each element `s` in `tuple_var`, emit `bits(s, width)`.

When the total domain size exceeds `--max-materialized-examples` (default `65536` for solving), `sat_synth_cpp` avoids unrolling the truth table into RAM and automatically switches to symbolic QF_BV CEGIS (`SymbolicVerifier`).

## Layout

- `sat_synth_cpp.cpp`: thin `main()` entrypoint delegating to `run_cli`.
- `cli.*`: `cxxopts` command-line parsing, `run_cli` orchestration, and user-facing logging.
- `config.*`: JSON config loading and dataset dumping.
- `datasets.*`: loader and evaluator for declarative dataset config files in `cpp/datasets/*.json`.
- `datasets/*.json`: declarative quantified dataset definitions.
- `quantified_spec.hpp`: declarative AST for quantified domains, predicates, and basis seeds.
- `symbolic_verifier.*`: Z3 QF_BV symbolic miter verification, counterexample extraction, and candidate checking.
- `program.*`: program/data model, operator table, text emission, and packed verifier.
- `postprocess.*`: post-processing beam orchestration and public options.
- `postprocess_common.*`, `postprocess_mask.*`, `postprocess_resynthesis.*`:
  scoring, shared candidate utilities, mask-only simplification, and local SAT
  resynthesis.
- `encoding.*`: Z3 structure/example encoding, solver construction, and model extraction.
- `solver.*`: solver orchestration, CEGIS, batching, timing, and verification result packaging.
- `tests/`: modular Catch2 unit and end-to-end CLI integration tests (`integration_tests.cpp`).

## Run

```bash
build/release/sat_synth_cpp --config path/to/config.json --cegis --cegis-initial-size 1
build/release/sat_synth_cpp --dataset adder --cegis
build/release/sat_synth_cpp --config typed-dataset.json --dump-dataset
build/release/sat_synth_cpp --config path/to/config.json --assume path/to/program.txt
build/release/sat_synth_cpp --config path/to/config.json --make-smt2
build/release/sat_synth_cpp --config path/to/config.json --make-dimacs
build/release/sat_synth_cpp --config path/to/config.json --make-blif
build/release/sat_synth_cpp --config path/to/config.json --output-blif
build/release/sat_synth_cpp --config path/to/config.json --post-process --post-process-resynthesis-maxnodes 5
build/release/sat_synth_cpp --config path/to/config.json --post-process --post-process-score 'program-length;max-output-depth,operator-cost'
build/release/sat_synth_cpp --config path/to/config.json --post-process --post-process-replace-patience 50
build/release/sat_synth_cpp --config path/to/config.json -vv
build/release/sat_synth_cpp --config path/to/config.json --profile
build/release/sat_synth_cpp --list-datasets
```

The emitted program uses the text format:

```text
T0: XOR(I0, I1)
OUT0: T0
```

Assumption files use the same text format. Blank lines and lines starting with `#` are ignored. Use `--assume -` to read assumptions from stdin.

Human-readable progress logs go to stderr. By default, the solver prints high-level lifecycle messages. Use `-v`/`--verbose` for detailed solver progress and repeat it, for example `-vv`, for trace-level batch and post-processing round summaries. `--quiet` suppresses these progress logs but does not suppress errors or `--profile`.

`--profile` prints phase timings and counters to stderr. It covers dataset generation, structure encoding, example packing/encoding, Z3 solve time, model extraction, post-processing, per-generator post-processing counters, packed verification, symbolic verification, and Z3 bit-vector literal cache hits/misses.

`--post-process` runs deterministic simplifications after model extraction and before text or BLIF emission. It can remove unreachable instructions, redirect equivalent masks to earlier sources, apply simple `AND`/`OR`/`XOR` algebra, and simplify output selectors under don't-care masks. It also tries structural regrouping/bypass generators, deterministic  eplacement search, and local SAT resynthesis for closed one-fanout windows. Use `--post-process-replace-patience`, `--post-process-resynthesis-maxnodes`, `--post-process-resynthesis-patience`, and `--generator-timeout` to control that search.

`--post-process-score` selects lexicographic score metrics. Separate metrics in one phase with commas and phases with semicolons. Prefix a metric with `-` for descending order. Supported metrics are `program-length`, `output-depth`, `max-output-depth`, `sum-output-depth`, `total-node-depth`, `total-tree-size`, `operator-cost`, `xor-count`, `output-cone-size`, `max-output-cone-size`, `sum-output-cone-size`, `fanout`, `max-fanout`, `sum-fanout`, `one-fanout-count`, `independent-pairs`, `entropy`, and `random`. The C++ `random` metric is deterministic for a fixed `--seed`.

## Test

```bash
ctest --test-dir build/debug --output-on-failure
```

By default, CTest runs the native Catch2 test suite (`cpp_unit_tests`), which includes unit tests for all modules as well as end-to-end CLI integration tests (`cpp/tests/integration_tests.cpp`), with zero Python runtime dependency.
