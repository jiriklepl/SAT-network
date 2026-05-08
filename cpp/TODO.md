# C++ TODOs

## Near Term

- Add a `solver.*` module so `sat_synth_cpp.cpp` only handles CLI orchestration.
- Introduce a structured `SolveResult` with status, program, elapsed time, and diagnostics.
- Add focused C++ unit-style tests for:
  - `program.*` packed verification and source formatting
  - `config.*` explicit config parsing and dataset dumping
  - `datasets.*` generated dataset parity
  - `cli.*` option parsing and validation
- Add CI coverage for CMake configure/build and `test_cpp_solver.py`.
- Decide the supported compiler baseline and document it.

## Dataset Parity

- Add exact Python-vs-C++ dataset comparison coverage for every supported default dataset.
- Decide how to handle state-space `max_examples` sampling:
  - either reproduce Python's sampling exactly, or
  - keep it unsupported and document the difference clearly.
- Add compact fixture configs for expensive full state-space datasets.
- Keep `--dump-dataset` stable enough for fixture-based regression tests.

## Solver Feature Parity

- Add `--assume` support.
- Add `--make-smt2`.
- Add `--make-dimacs`.
- Add `--make-blif`.
- Add `--output-blif`.
- Keep quantified synthesis deferred until the explicit-example path is fully stable.
- Keep post-processing deferred until solver/export parity is validated.

## Performance

- Profile phases separately:
  - dataset generation
  - example packing
  - structure encoding
  - example encoding
  - Z3 solve time
  - model extraction
  - packed verification
- Cache common Z3 constants and selector values.
- Avoid copying example batches where spans or index ranges are enough.
- Consider a word-vector packed mask representation before falling back to `cpp_int`.
- Measure whether balanced selectors, boolean encoding, and tactic choices should have dataset-specific defaults.

## Post-Processing Port

- Start with pure mask-based simplifications that do not require Z3.
- Port local SAT resynthesis as a separate module after mask-only transforms are tested.
- Add fixtures that compare Python and C++ post-processed program behavior, not textual identity.
- Keep scoring and candidate generation modular so individual generators can be tested independently.

## Developer UX

- Add CMake presets for debug, release, and sanitizer builds.
- Add dependency setup notes for common Linux/macOS environments.
- Add formatting/linting guidance if the C++ surface keeps growing.
- Keep `cpp/README.md` current as module boundaries change.
