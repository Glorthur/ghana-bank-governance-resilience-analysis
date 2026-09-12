# Reproducibility

The package is reproducible from the frozen verified input at `data/frozen/bank_year_master.csv`. The six-year, 16-bank panel has unique bank-year keys; the primary analysis retains 95 complete observations. The sole primary exclusion is First Atlantic Bank PLC, 2024, which lacks the ORI and governance evidence required for the core model and is not imputed.

## Workflow

1. Create a Python 3.12 environment and install `requirements.txt`.
2. Run `scripts/run_analysis.py` with the frozen CSV and an output directory.
3. Run `scripts/second_pass_validation.py` and `scripts/audit_outputs.py` against the generated artifacts.
4. Run `pytest`.

The scripts record input hashes and generate tables, figures, diagnostics, logs, and manifests. Numerical checks should use absolute tolerance 1e-8; image bytes need not be identical across platforms.

## What is and is not reproduced

The workflow reproduces cleaning checks, formula-derived variables, sample selection, fixed-effects estimation, clustered and wild-cluster inference, robustness specifications, and validation of generated artifacts. It starts after upstream annual-report collection and manual coding. Source PDFs, extracted text, private Drive paths, and the thesis DOCX are intentionally excluded. The Digital Garden is not edited by this repository workflow.
