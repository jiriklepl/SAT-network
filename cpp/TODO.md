# C++ TODOs

## Near Term

- Continue expanding Catch2 edge-case coverage as new C++ modules and features are added.
- Maintain GitHub Actions coverage as C++ dependencies and test targets change.
- Add macOS CI once the Linux C++ baseline is stable.

## Dataset Parity

- Optionally implement Python-compatible `max_examples` sampling later.
- Add more compact fixture configs when new parametric datasets are introduced.
- Keep `--dump-dataset` stable enough for fixture-based regression tests.

## Solver Feature Parity

- Keep quantified synthesis deferred until the explicit-example path is fully stable.
- Keep post-processing deferred until solver/export parity is validated.

## Post-Processing Port

- Start with pure mask-based simplifications that do not require Z3.
- Port local SAT resynthesis as a separate module after mask-only transforms are tested.
- Add fixtures that compare Python and C++ post-processed program behavior, not textual identity.
- Keep scoring and candidate generation modular so individual generators can be tested independently.

## Developer UX

- Add sanitizer CMake presets.
- Add dependency setup notes for common Linux/macOS environments.
- Add formatting/linting guidance if the C++ surface keeps growing.
- Keep `cpp/README.md` current as module boundaries change.
