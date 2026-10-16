# C++ TODOs

## Near Term

- Optionally add sanitizer CI coverage once local sanitizer runs are stable enough.
- Add macOS CI once the Linux C++ baseline is stable.

## Dataset Parity

- Add more compact fixture configs when new parametric datasets are introduced.
- Keep `--dump-dataset` stable enough for fixture-based regression tests.

## Solver Feature Parity

- Keep quantified synthesis deferred until the explicit-example path is fully stable.
- Maintain Python-compatible multi-phase post-process scoring as new metrics or candidate generators are added.

## Post-Processing Port

- Extend mask-only simplification coverage as new algebraic opportunities are found.
- Add fixtures that compare Python and C++ post-processed program behavior, not textual identity.
- Maintain the modular candidate generator layout as additional Python-only strategies are evaluated.

## Developer UX

- Add dependency setup notes for common Linux/macOS environments.
- Add formatting/linting guidance if the C++ surface keeps growing.
- Keep `cpp/README.md` current as module boundaries change.
