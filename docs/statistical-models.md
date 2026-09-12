# Statistical models

## Estimand and sample

The principal estimand is the within-bank association between governance measures and the ORI disclosure score after controlling for common year shocks. The primary complete sample has 95 observations from 16 banks over 2020–2025. The ORI takes observed non-missing values 62.5, 75, 87.5, and 100; 51 of 95 observations (53.7%) equal 100.

## Principal specifications

For bank (i) and year (t), the direct-effects model is:

`ORI_it = β1 board_independence_c_it + β2 firm_size_c_it + β3 ROA_it + β4 liquidity_it + β5 leverage_it + α_i + γ_t + ε_it`

The moderation model adds `β6 board_independence_c_it × firm_size_c_it`. Board independence and log assets are grand-mean centred before forming the interaction. `α_i` are bank fixed effects and `γ_t` are year fixed effects. Inference uses standard errors clustered by bank; wild-cluster bootstrap p-values are supplied because there are only 16 clusters.

H1 is nondirectional. H2 predicts a positive firm-size effect, and H3 predicts a positive interaction. The audited estimates are H1 = -18.7602, H2 = 9.1332, and H3 = 14.3390. H1 is supported at 5% under the planned clustered and wild-cluster inference; H2 and H3 are not statistically supported.

## Model-selection and diagnostics

The Hausman p-value is 0.9994, while the Mundlak p-value is 0.0368, indicating correlation between bank effects and regressors and supporting FE for the substantive within-bank question. Heteroskedasticity is detected (p=0.0012); the Wooldridge serial-correlation test is p=0.0751 and Pesaran CD is p=0.7749.

## Sensitivity models

CAR is a reduced-sample robustness control (70 observations, 15 banks). Full-disclosure binary and fractional-response models address the bounded, ceiling-heavy outcome. NPL is a separate reduced-sample outcome (43 observations, 10 banks), not an ORI proxy. These models do not replace the pre-specified linear FE analysis. Nonlinear bank-indicator models are interpreted cautiously because the short panel can create incidental-parameter and separation problems.
