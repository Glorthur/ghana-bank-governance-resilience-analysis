# Chapter Four Drafting Outline

## 4.1 Introduction

State that the chapter reports the verified 2020-2025 bank panel, descriptive evidence, panel-model diagnostics, hypothesis tests, and bounded-outcome/reduced-sample robustness checks.

## 4.2 Sample and data quality

- Intended panel: 96 bank-years from 16 banks.
- Primary complete-case sample: 95 bank-years.
- Explain the single excluded observation using `tables/analysis_sample_register.csv`.
- Report CAR and NPL availability separately.
- State that ORI equals 100 in 51 observations and discuss the ceiling limitation before modelling.

Recommended exhibits: sample-flow table, missingness table, and ORI distribution figure.

## 4.3 Descriptive statistics

Use `tables/table_01_descriptive_core.csv`, `table_03_descriptive_by_year.csv`, and `table_06_ori_component_prevalence.csv`. Describe magnitude and variation without interpreting correlations as effects.

## 4.4 Correlations and preliminary diagnostics

Use the correlation matrix, VIF table, and diagnostics in `diagnostics/`. Explain why centred predictors are used in the moderation model.

## 4.5 Model choice

Present pooled OLS, fixed effects, and random effects side by side. Report both Hausman and Mundlak results. Identify two-way fixed effects with bank-clustered standard errors as the principal specification.

## 4.6 Hypothesis tests

### 4.6.1 Board independence and ORI (H1)

Report the baseline direct-effects FE coefficient -18.7602, clustered p-value 0.0238, and wild-cluster p-value 0.0392. Conclude: **supported at the 5% level**. State explicitly that the estimated association is negative.

### 4.6.2 Firm size and ORI (H2)

Report the baseline direct-effects FE coefficient 9.1332, clustered p-value 0.2398, and wild-cluster p-value 0.1744. Conclude: **not supported statistically**.

### 4.6.3 Moderating effect of firm size (H3)

Report the interaction coefficient 14.3390, clustered p-value 0.4413, and wild-cluster p-value 0.5447. Conclude: **not supported statistically**. Follow with the simple-slopes table and marginal-effects figure.

## 4.7 Robustness checks

Present CAR, full-disclosure, fractional-response, leave-one-bank-out, and NPL analyses as sensitivity evidence. Keep the 95-row FE model visibly primary and label every reduced sample.

## 4.8 Discussion

Relate only statistically supported patterns to agency, resource-dependence, and contingency theory. Discuss unsupported hypotheses directly. Balance governance benefits against information asymmetry, competence constraints, larger-bank complexity, cyberattack surface, legacy systems, and bureaucratic friction.

## 4.9 Chapter conclusion

Summarise the evidence without causal language. Carry the ceiling effect, small panel, and disclosure-versus-realised-resilience distinction into Chapter Five.
