# Ghana Bank Governance and Resilience Analysis

Reproducible analysis of governance and operational-resilience disclosure in a panel of Ghanaian banks, 2020–2025. The repository contains the analysis code, a curated frozen dataset, non-confidential provenance evidence, and the outputs needed to inspect the reported results.

## Scope and headline results

The primary sample contains 95 complete bank-years from 16 banks. The principal specification is a two-way (bank and year) fixed-effects model with bank-clustered standard errors. The outcome is an annual-report operational-resilience disclosure index (ORI), not a direct measure of realised recovery performance.

The baseline estimate for board independence is -18.7602 (clustered p=0.0238; wild-cluster p=0.0392). Firm size is 9.1332 (p=0.2398), and the board-independence × firm-size interaction is 14.3390 (p=0.4413; wild-cluster p=0.5447). These are associations, not causal estimates. See [statistical models](docs/statistical-models.md) and [limitations](docs/limitations.md).

## Reproduce

Use Python 3.12 in a clean environment, install `requirements.txt`, and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_analysis.py --input data/frozen/bank_year_master.csv --output outputs
python scripts/second_pass_validation.py --input data/frozen/bank_year_master.csv --output outputs
python scripts/audit_outputs.py --output outputs
```

The exact options may be inspected with `python scripts/<name>.py --help`. The scripts use repository-relative paths when options are omitted and write generated tables, figures, diagnostics, logs, and manifests below the selected output directory. Run the test suite with `pytest`.

## Reproducibility boundary

The reproducible boundary begins with the committed, frozen verified CSV in `data/frozen/`. It reproduces validation, derived variables, model estimation, diagnostics, and reported artifacts from that input. It does not reproduce the upstream extraction of annual reports, and it does not fetch or modify the private Drive archive, source PDFs, thesis manuscript, or Digital Garden. Provenance and coding decisions are documented in `docs/`.

## Repository map

- `data/`: curated frozen data and provenance evidence.
- `scripts/`: portable analysis, validation, and audit entry points.
- `docs/`: methodology, model, provenance, reproducibility, limitations, and handoff notes.
- `outputs/`: generated artifacts after running the analysis.

## Attribution and licensing

Research, analysis design, verification, and repository curation: Gloria Arthur. Authorship details are in [AUTHORS.md](AUTHORS.md), and citation metadata is in [CITATION.cff](CITATION.cff). Code is MIT licensed; the original documentation and curated/derived data are CC BY 4.0 licensed. See [LICENSE-CODE](LICENSE-CODE) and [LICENSE-DATA](LICENSE-DATA).
