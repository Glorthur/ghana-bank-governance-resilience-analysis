# Second-Pass Statistical Validation

## Bottom line

The second pass confirms support for the nondirectional H1 in the baseline two-way FE model: the board-independence association is statistically significant and negative under both bank-clustered and wild-cluster inference. H2 and H3 remain unsupported.

The same six substantive coefficients were reproduced with three independent calculations: `linearmodels` two-way FE, a statsmodels least-squares dummy-variable regression, and a manual Frisch-Waugh-Lovell residualisation. The largest pairwise coefficient difference was **2.807e-13**.

## Primary hypothesis audit

| Hypothesis | FE coefficient | Clustered SE | 95% CI | Clustered p | Wild-cluster p | Bank-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| H1 | -18.7602 | 8.1147 | -34.6650 to -2.8554 | 0.0238 | 0.0392 | -33.7823 to 0.4085 |
| H2 | 9.1332 | 7.7022 | -5.9631 to 24.2296 | 0.2398 | 0.1744 | -8.3379 to 29.9017 |
| H3 | 14.3390 | 18.5118 | -21.9441 to 50.6220 | 0.4413 | 0.5447 | -32.8367 to 48.7780 |

The confidence intervals include zero for all three hypotheses. H1 is negative in the principal model; H2 and H3 are positive but imprecisely estimated.

## Why the mixed findings are plausible

- Only **10 of 16 banks** show any within-bank ORI variation.
- The within-bank ORI standard deviation is **7.357**, while 51 of 95 observations sit at the maximum score of 100.
- Board independence and firm size vary within all 16 banks, but the outcome supplies limited within-bank information for estimating their effects.
- The H3 clustered standard error is **18.512**, almost twice the point estimate, so the interaction is weakly identified.
- With only 16 clusters, conventional significance can be optimistic; the 9,999-draw wild-cluster test nevertheless confirms H1 at 5%, while H2 and H3 remain above 0.05.

## Specification checks

The coefficient point estimates are invariant to covariance choice. Unadjusted, heteroskedastic-robust, bank-clustered, and Driscoll-Kraay standard errors change uncertainty, not the underlying two-way FE coefficients.

- Heteroskedasticity-only standard errors make H2 significant at 5%, but they ignore repeated observations within banks and are not the planned inference method.
- Driscoll-Kraay standard errors make all three coefficients significant, but this estimator relies on a sufficiently long time dimension. With only **T=6** and bandwidth 2, its very small standard errors are not credible for this panel and should not be used for hypothesis decisions.
- Bank-clustered errors directly address within-bank dependence, and the wild-cluster bootstrap addresses the small number of banks. Both support the nondirectional H1 and leave H2-H3 unsupported.
- The main-effects FE model without the interaction is the stated direct-effects model for H1 and H2; the moderation model remains the stated test for H3.
- Removing year effects, using random effects, and using pooled OLS answer different identifying questions. The significant pooled H1/H3 results disappear after controlling for time-invariant bank effects, consistent with the earlier Mundlak finding.

H1 is evaluated with a two-sided test because it is nondirectional. H2 and H3 retain their positive theoretical expectations; neither reaches 0.05 under directional or two-sided inference.

## Bounded-outcome checks with bank indicators

- Binary full-disclosure GLM with bank and year indicators: H3 coefficient -0.0667, p=0.9945.
- Fractional logit GLM with bank and year indicators: H3 coefficient 2.1407, p=0.3946.
- Poisson missing-disclosures GLM with bank and year indicators: H3 coefficient -1.8970, p=0.3815.

These nonlinear bank-indicator models are sensitivity analyses. With at most six observations per bank, they can suffer incidental-parameter bias and must not replace the pre-specified linear FE model solely because one p-value crosses 0.05.

The binary full-disclosure model exhibits non-finite standard errors and p-values, consistent with separation caused by banks that are always at or always below full disclosure. It is not usable for inference. The fractional and Poisson bank-indicator checks yield nonsignificant H3 estimates and reinforce the primary conclusion.

## Data and coding audit

- Frozen input hash is available for the supplied input: **True**.
- ORI, board independence, firm size, ROA, liquidity, and leverage all recompute exactly from their source columns.
- The primary sample remains 95 observations from 16 banks.
- The only core exclusion remains First Atlantic Bank PLC, 2024; it is not imputed.
- The interaction uses grand-mean-centred board independence and firm size exactly as stated in the methodology.

## Interpretation

The corrected conclusion is that the data support a statistically significant negative within-bank association between board independence and operational-resilience disclosure under nondirectional H1. H2 and H3 remain unsupported. The ceiling-heavy ORI and short six-year panel still limit statistical power, especially for moderation.
