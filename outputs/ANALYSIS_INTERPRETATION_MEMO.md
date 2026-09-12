# Analysis Interpretation Memo

## Executive finding

The staged fixed-effects analysis uses **95 bank-year observations from 16 banks**, with bank and year effects and bank-clustered standard errors. H1 and H2 are evaluated in the baseline direct-effects model, while H3 is evaluated in the moderation model. The outcome is highly ceiling-concentrated: **51 of 95 observations (53.7%) score 100**. Results therefore describe within-bank changes in a disclosure index and must not be framed as causal evidence of realised operational resilience.

The baseline FE model produces a statistically significant negative board-independence coefficient of **-18.7602 (p=0.0238)**. Under the nondirectional H1, this supports the existence of a within-bank association, but its negative sign must be reported and explained. The moderation model's interaction coefficient is **14.3390 (p=0.4413)** and is not statistically significant.

## Hypothesis results

| Hypothesis | FE coefficient | Clustered p-value | Wild-cluster p-value | Decision |
|---|---:|---:|---:|---|
| H1: Board independence is associated with ORI | -18.7602 | 0.0238 | 0.0392 | supported at the 5% level |
| H2: Firm size positively affects ORI | 9.1332 | 0.2398 | 0.1744 | not supported statistically |
| H3: Board independence x firm size -> ORI | 14.3390 | 0.4413 | 0.5447 | not supported statistically |

H1 and H2 are direct effects from the baseline FE model. The H3 interaction and the simple slopes below come from the moderation FE model.

## Moderation interpretation

The interaction estimate is **14.3390**. Its sign is retained in **93.8%** of the 16 leave-one-bank-out specifications, but it is significant at 5% in **1 of 16**. The simple-slope results are:

| firm_size_level   |   firm_size_c |   board_independence_slope |   std_error |   ci_lower_95 |   ci_upper_95 |   p_value | significance   |
|:------------------|--------------:|---------------------------:|------------:|--------------:|--------------:|----------:|:---------------|
| Firm size: -1 SD  |     -0.610902 |                  -24.0851  |    10.5021  |      -44.6692 |      -3.50097 | 0.0249244 | **             |
| Firm size: mean   |      0        |                  -15.3254  |     8.36233 |      -31.7156 |       1.06475 | 0.0712295 | *              |
| Firm size: +1 SD  |      0.610902 |                   -6.56573 |    16.8921  |      -39.6743 |      26.5429  | 0.698723  |                |

Use `figures/figure_03_marginal_effect_board_independence.png` when discussing H3. Emphasise the confidence interval and avoid claiming moderation where it overlaps zero.

## Model selection

- Hausman p-value: **0.9994**. Do not reject RE consistency.
- Mundlak p-value: **0.0368**. Bank effects are correlated with regressors; FE is preferred.
- Fixed effects remain the principal model because the research question concerns within-bank change and unobserved time-invariant bank characteristics are substantively plausible.

## Diagnostics

- Heteroskedasticity test p-value: **0.0012**.
- Wooldridge serial-correlation p-value: **0.0751**.
- Pesaran CD p-value: **0.7749**.
- Bank-clustered inference is used throughout the linear panel models. Wild-cluster bootstrap p-values are supplied for H1-H3 because there are only 16 clusters.

## Robustness boundaries

- CAR model: estimated. It uses a reduced sample and is not co-primary.
- Full-disclosure sensitivity: estimated.
- Fractional-response sensitivity: estimated.
- NPL model: estimated; separate outcome, not an ORI proxy. NPL is a separate outcome and must not be described as an ORI proxy.

The CAR-augmented FE interaction is **16.8640 (p=0.4038)**. The year-adjusted pooled binary full-disclosure interaction is **2.6062 (p=0.2641)**, and the year-adjusted pooled fractional-response interaction is **0.6692 (p=0.5243)**. The fractional specification is significant, but it does not include bank fixed effects and conflicts with the principal FE result and wild-cluster inference. It is therefore sensitivity evidence, not sufficient support for H3.

## Required caveats

1. The panel is small: 16 banks over at most six years.
2. ORI takes the observed non-missing values 62.5, 75, 87.5, 100, with a strong ceiling effect.
3. The annual-report index measures disclosure evidence, not directly observed disruption recovery performance.
4. CAR and NPL results rely on substantially smaller, non-randomly available samples.
5. Associations are not causal estimates.
6. Do not add bank age unless a complete, independently verified bank-year measure is assembled.
