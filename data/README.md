# Curated data

This directory contains the non-confidential, curated data boundary for the Ghana bank governance and resilience analysis. No source PDFs, extracted report text, or machine-local paths are included.

## Files

- `frozen/bank_year_master.csv` — audited frozen panel used as the reproducibility input: 16 banks observed for 2020–2025 (96 bank-years).
- `processed/analysis_ready_panel.csv` — current release of the analysis-ready panel, including derived sample flags, centered covariates, interaction terms, and disclosure measures.
- `metadata/data_dictionary.csv` — variable definitions, units, and coding/formulas.
- `metadata/source_register.csv` — report-level provenance with stable `source_id` values, hashes, and extraction/verification metadata. Local paths and extracted-text filenames are intentionally omitted.
- `evidence/ori_evidence.csv` — row/item-level ORI coding evidence and adjudication fields.
- `evidence/manual_verification_register.csv` — manual verification and correction register.

The frozen panel preserves audited substantive values. Monetary values are absolute Ghana cedis; reported ratios retain their documented units. Treat the frozen panel as the input to the portable analysis scripts.
