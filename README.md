# Ghana Bank Governance and Resilience Analysis

This repository presents a reproducible analysis of corporate governance and operational-resilience disclosure among Ghanaian banks between 2020 and 2025. It contains the analysis code, a curated frozen dataset, non-confidential provenance records, and the outputs required to inspect and reproduce the reported results.

## Scope and main findings

The primary sample consists of 95 complete bank-year observations from 16 banks. The principal specification is a two-way fixed-effects model, controlling for both bank and year effects, with standard errors clustered at the bank level.

Operational resilience is measured using an annual-report disclosure index (ORI). The index captures the extent to which banks disclose selected operational-resilience practices. It should not be interpreted as a direct measure of realised recovery performance or a bank's actual ability to withstand disruption.

In the baseline model, board independence has an estimated coefficient of `-18.7602`, with a clustered p-value of `0.0238` and a wild-cluster p-value of `0.0392`. The coefficient for firm size is `9.1332` (`p = 0.2398`), while the interaction between board independence and firm size is `14.3390` (`p = 0.4413`; wild-cluster `p = 0.5447`).

These results indicate statistical associations within the observed panel. They do not establish causal relationships. The modelling decisions and important qualifications are discussed in the [statistical models](docs/statistical-models.md) and [limitations](docs/limitations.md) documents.

## Reproducing the analysis

Use Python 3.12 in a clean virtual environment. On Windows PowerShell, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python scripts/run_analysis.py `
  --input data/frozen/bank_year_master.csv `
  --output outputs

python scripts/second_pass_validation.py `
  --input data/frozen/bank_year_master.csv `
  --output outputs

python scripts/audit_outputs.py `
  --input data/frozen/bank_year_master.csv `
  --output outputs

python -m pytest
```

Each script also supports repository-relative defaults, so the arguments may be omitted during normal use. To inspect the available options, run:

```powershell
python scripts/run_analysis.py --help
python scripts/second_pass_validation.py --help
python scripts/audit_outputs.py --help
```

The pipeline writes the generated tables, figures, diagnostic results, logs, and audit manifests to the selected output directory.

## Reproducibility boundary

Reproduction begins with the committed and verified dataset in `data/frozen/`. From this input, the scripts reproduce the data checks, derived variables, statistical models, diagnostics, tables, and figures.

The repository does not reproduce the earlier extraction of information from annual reports. It also does not retrieve or modify the private Drive archive, original source PDFs, thesis manuscript, supervisor comments, or Digital Garden. The relevant provenance and coding decisions are documented in `docs/`.

## Repository structure

- `data/` contains the frozen dataset, processed panel, metadata, and non-confidential provenance evidence.
- `scripts/` contains the portable analysis, validation, and auditing programs.
- `docs/` explains the methodology, ORI coding process, statistical models, data provenance, reproducibility boundary, and limitations.
- `outputs/` contains the reproduced tables, figures, diagnostics, logs, and manifests.
- `tests/` contains the checks used to validate the data, analytical samples, regression outputs, and repository contents.

## Attribution and licensing

Gloria Arthur conducted the research, designed and verified the analysis, curated the data, and developed the reproducibility repository. Further contribution details are provided in [AUTHORS.md](AUTHORS.md), while the recommended citation information is available in [CITATION.cff](CITATION.cff).

The original code is licensed under the [MIT License](LICENSE-CODE). The original documentation and curated or derived data are licensed under [CC BY 4.0](LICENSE-DATA). Third-party annual reports, academic articles, thesis drafts, and supervisor comments are not included or relicensed.
