# Data Validation Report

## Overall assessment: Ready for analysis with stated caveats

- Intended rows: 96
- Banks: 16
- Primary complete observations: 95
- CAR observations: 70
- NPL observations: 43
- ORI observations at 100: 51

All identity, range, uniqueness, and recomputed formula checks passed.

The single primary-sample exclusion is First Atlantic Bank PLC, 2024, because the available source is a financial-only summary without the ORI and governance evidence required for the core model. It is not imputed.

All CAR and NPL exclusions are documented bank-year by bank-year in `tables/analysis_sample_register.csv`.
