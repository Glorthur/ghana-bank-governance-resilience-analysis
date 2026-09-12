# Data methodology

## Reproducibility boundary

The public analysis begins with `data/frozen/bank_year_master.csv`, a verified 16-bank, six-year panel covering 2020–2025. The upstream annual reports were treated as read-only research materials. This repository publishes derived tabular values and non-confidential provenance evidence, not report PDFs or extracted report text.

## Panel construction

Each row identifies a bank-year using `bank_id`, `bank`, and `year`. Financial statement values were transcribed and manually checked from annual-report disclosures. Monetary values were normalized to absolute Ghana cedis. Ratios reported by banks (CAR and NPL) remain percentages; calculated rates such as ROA, liquidity, leverage, and board independence are proportions. `firm_size_ln_assets` is the natural logarithm of total assets.

The frozen panel is the authoritative input. The current-release panel adds analysis-ready variables and flags while retaining the source measures. Missingness and sample membership are represented in the supplied columns; downstream scripts must not silently substitute values.

## Verification

Report identity, year, hashes, extraction status, and review outcomes are recorded in `data/provenance/source_register.csv`. Manual financial, board, and ORI review outcomes are recorded in `manual_verification_register.csv`. The item-level ORI evidence register provides coding and adjudication traceability without exposing local storage locations.
