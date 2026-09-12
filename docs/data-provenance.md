# Data provenance

## Stable identifiers

Every annual-report observation is linked to a stable identifier of the form `annual_report:BANK_ID:YEAR`, for example `annual_report:ABSA_BANK_GHANA_PLC:2020`. `BANK_ID` is the uppercase normalized identifier in the frozen panel. The source register also records the report filename and SHA-256 hash so that an authorized custodian can verify the underlying source independently.

## Published provenance

`data/metadata/source_register.csv` contains report-level metadata, including page counts, searchable-page measures, extraction status, report status, hashes, and manual adjudication notes. Local absolute paths and extracted-text paths have been removed for portability and privacy.

`data/evidence/ori_evidence.csv` contains the eight-item evidence trail for each bank-year, including page references, matched coding rules, concise evidence excerpts, pass comparison, and final adjudication. `data/evidence/manual_verification_register.csv` records the review status, corrections, review date, and notes for each bank-year.

The provenance files are evidence about the curated dataset; they are not a substitute for the underlying source reports. Source reports and extracted text remain outside this repository.

## Release boundary

The frozen CSV is the audited reproducibility input. The processed CSV is an analysis-ready derivative and may contain additional computed columns. Any future data release should preserve the frozen file, update hashes and review metadata, and document changes rather than overwriting the historical boundary.
