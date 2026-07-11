# Changelog

## 0.2.0

AFISP is now a pure-Python, pip-installable package. **R is no longer required.**

### Changed
- **Removed the R dependency.** The `method="SIRUS"` path now uses the
  pure-Python [`sirus`](https://github.com/asubbaswamy/sirus-py) package in
  process instead of shelling out to `Rscript`. Installation is now a single
  `pip install afisp` — no conda environment, no `r-base`, no manual
  `Rscript install_R_packages.R` step.
- **`SubgroupPhenotyper.fit` now defaults to `method="SIRUS"`** (previously
  `"DecisionList"`), since SIRUS installs friction-free.
- Added a `random_state` argument to `SubgroupPhenotyper.fit` for reproducible
  SIRUS fits. Note that the Python `sirus` package is an independent
  re-implementation of the R package: a fixed seed makes runs reproducible but
  does not reproduce the R package's exact rules.
- `subgroup_feature_data` inputs are now validated for `method="SIRUS"`: all
  columns must be numeric (encode categoricals as dummy variables) and
  `subset_labels` must contain exactly two classes.

### Removed
- **Breaking:** removed the `input_fname` / `output_fname` arguments from
  `SubgroupPhenotyper.fit` (they only named R temp files).
- Deleted the R toolchain: `afisp/run_sirus.r`, `install_R_packages.R`,
  `R_packages.rda`, `environment.yml`, and `pip_packages.txt`.

### Added
- `pyproject.toml` (pip-installable package, `requires-python >= 3.10`), with a
  `torch` extra for the optional `torch_roc_auc_surrogate` loss.
- A `pytest` test suite under `tests/` and a GitHub Actions CI workflow.

### Notes on behavior parity
- The default `method="SIRUS"` call (`cv=False`, `p0=0.025`) preserves the
  historical rule cap of 25 (the original R script did not pass `num.rule.max`
  in its explicit-`p0` branch). Because SIRUS is an independent re-implementation
  of the R package, extracted rule strings may differ slightly from the R output
  while remaining statistically equivalent.
