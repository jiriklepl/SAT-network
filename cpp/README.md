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
  "type": "adder",
  "params": {
    "num_inputs": 4
  },
  "constraints": [
    {
      "check": "$num_inputs >= 2 and $num_inputs % 2 == 0",
      "error": "adder num_inputs must be an even integer >= 2"
    }
  ],
  "let": {
    "half": "$num_inputs // 2",
    "out_bits": "$half + 1"
  },
  "instructions": "3 * $half",
  "forall": [
    { "name": "a", "type": "range", "stop": "1 << $half" },
    { "name": "b", "type": "range", "stop": "1 << $half" }
  ],
  "let_row": [
    { "name": "total", "value": "$a + $b" }
  ],
  "where": "true",
  "inputs": {
    "concat": [
      { "bits_be": ["$a", "$half"] },
      { "bits_be": ["$b", "$half"] }
    ]
  },
  "outputs": {
    "bits_be": ["$total", "$out_bits"]
  }
}
```

### Schema Reference

- **`params`** *(object, optional)*: Default parameter dictionary. Can be overridden at the top level of a user config or nested under a section matching `"type"`.
- **`constraints`** *(array, optional)*: Static validation rules `{"check": "<expr>", "error": "<message>"}` evaluated before building the dataset.
- **`let`** *(object, optional)*: Static constants evaluated in lexicographical/dependency order before quantification.
- **`instructions`** *(integer or expression string, optional)*: Default SSA instruction budget (e.g. `14` or `"3 * $half"`).
- **`forall`** *(array of domains)*: Quantified variables spanning the input domain:
  - `{"name": "x", "type": "bits", "width": "<expr>"}`: Bit-tuple of length `width` (`0`/`1` elements).
  - `{"name": "x", "type": "tuples", "length": "<expr>", "base": "<expr>"}`: Tuple of `length` integers in `[0, base - 1]`.
  - `{"name": "x", "type": "range", "start": "<expr>", "stop": "<expr>"}`: Integer in `[start, stop - 1]` (`start` defaults to `0`).
- **`let_row`** *(array of `{"name": "<id>", "value": <expr>}`, optional)*: Per-row intermediate bindings evaluated in array order; each binding can reference `$forall` variables and earlier `let_row` variables.
- **`where`** *(expression, optional)*: Boolean domain filter predicate. Only assignments where `where` evaluates to true are required to satisfy the input/output mapping.
- **`inputs`** / **`outputs`**: Bit-vector expressions (or arrays of boolean/`null` expressions) defining the circuit inputs and expected outputs (`null` denotes a don't-care output bit).

### Supported Expressions & Built-in Functions

- **Variables & Indexing**: Reference variables with `"$name"`, index tuples with `"$cells[0]"`, or use JSON objects `{"var": "name"}` and `{"index": [<list>, <idx>]}`.
- **Operators**:
  - Ternary conditional: `cond ? then_expr : else_expr`
  - Logical: `or`, `and`, `not`
  - Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
  - Bitwise: `|`, `^`, `&`, `<<`, `>>`
  - Arithmetic: `+`, `-`, `*`, `//` (floor division), `%` (modulo)
- **Built-in Functions**:
  - `sum(list)` / `popcount(int)`: Sum of tuple elements or 32-bit population count.
  - `count_eq(list, value)`: Count elements in `list` equal to `value`.
  - `bit(value, bit_index)`: Extract a single bit `(value >> bit_index) & 1`.
  - `bits_be(value, width)` / `bits_le(value, width)`: Expand integer `value` into `width` bits (big-endian MSB-first or little-endian LSB-first).
  - `int_be(list)` / `int_le(list)`: Pack a bit list into an integer.
  - `concat(list_of_lists)`: Flatten/concatenate bit lists.
  - `rotate_left_2x2(cells)`: Rotate a 4-cell $2 \times 2$ block `[a, b, c, d]` into `[b, d, a, c]`.

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
