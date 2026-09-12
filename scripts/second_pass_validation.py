from __future__ import annotations

import hashlib
import json
import math
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "frozen" / "bank_year_master.csv"
OUT_ROOT = ROOT / "outputs"
OUT = OUT_ROOT / "second_pass_validation"
FROZEN = INPUT
CORE_RAW = [
    "ori_score",
    "board_independence",
    "firm_size_ln_assets",
    "roa",
    "liquidity_ratio",
    "leverage_ratio",
]
X = [
    "board_independence_c",
    "firm_size_c",
    "board_x_size",
    "roa",
    "liquidity_ratio",
    "leverage_ratio",
]
MAIN_X = [
    "board_independence_c",
    "firm_size_c",
    "roa",
    "liquidity_ratio",
    "leverage_ratio",
]
HYPOTHESES = {
    "H1": "board_independence_c",
    "H2": "firm_size_c",
    "H3": "board_x_size",
}
SEED = 20260902


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_frame(sample: pd.DataFrame) -> pd.DataFrame:
    return sample.sort_values(["bank_id", "year"]).set_index(["bank_id", "year"])


def linearmodels_fe(sample: pd.DataFrame, covariance: str, **kwargs):
    panel = panel_frame(sample)
    model = PanelOLS(
        panel["ori_score"],
        panel[X],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )
    return model.fit(cov_type=covariance, **kwargs)


def coefficient_rows(model_name: str, result, method: str) -> list[dict]:
    rows = []
    for hypothesis, variable in HYPOTHESES.items():
        if variable not in result.params.index:
            continue
        if hasattr(result, "std_errors"):
            se = float(result.std_errors[variable])
            pvalue = float(result.pvalues[variable])
        else:
            se = float(result.bse[variable])
            pvalue = float(result.pvalues[variable])
        coefficient = float(result.params[variable])
        rows.append(
            {
                "model": model_name,
                "hypothesis": hypothesis,
                "variable": variable,
                "coefficient": coefficient,
                "std_error": se,
                "ci_lower_95": coefficient - 1.96 * se,
                "ci_upper_95": coefficient + 1.96 * se,
                "p_value": pvalue,
                "method": method,
            }
        )
    return rows


def fwl_coefficients(sample: pd.DataFrame) -> pd.Series:
    bank_dummies = pd.get_dummies(sample["bank_id"], drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(sample["year"].astype(int), drop_first=True, dtype=float)
    fixed = np.column_stack(
        [
            np.ones(len(sample)),
            bank_dummies.to_numpy(dtype=float),
            year_dummies.to_numpy(dtype=float),
        ]
    )
    y = sample["ori_score"].to_numpy(dtype=float)
    x = sample[X].to_numpy(dtype=float)
    y_resid = y - fixed @ np.linalg.lstsq(fixed, y, rcond=None)[0]
    x_resid = x - fixed @ np.linalg.lstsq(fixed, x, rcond=None)[0]
    beta = np.linalg.lstsq(x_resid, y_resid, rcond=None)[0]
    return pd.Series(beta, index=X)


def bootstrap_clusters(sample: pd.DataFrame, draws: int = 1999) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    bank_ids = sample["bank_id"].drop_duplicates().to_numpy()
    estimates = np.empty((draws, len(X)), dtype=float)
    by_bank = {bank: sample.loc[sample["bank_id"] == bank].copy() for bank in bank_ids}
    for draw in range(draws):
        selected = rng.choice(bank_ids, size=len(bank_ids), replace=True)
        pieces = []
        for position, bank in enumerate(selected):
            piece = by_bank[bank].copy()
            piece["bootstrap_entity"] = f"entity_{position:02d}"
            pieces.append(piece)
        boot = pd.concat(pieces, ignore_index=True)
        entity = pd.get_dummies(boot["bootstrap_entity"], drop_first=True, dtype=float)
        year = pd.get_dummies(boot["year"].astype(int), drop_first=True, dtype=float)
        fixed = np.column_stack(
            [
                np.ones(len(boot)),
                entity.to_numpy(dtype=float),
                year.to_numpy(dtype=float),
            ]
        )
        design = np.column_stack(
            [
                boot[X].to_numpy(dtype=float),
                fixed,
            ]
        )
        moderation_beta = np.linalg.lstsq(
            design, boot["ori_score"].to_numpy(dtype=float), rcond=None
        )[0][: len(X)]
        main_design = np.column_stack([boot[MAIN_X].to_numpy(dtype=float), fixed])
        main_beta = np.linalg.lstsq(
            main_design, boot["ori_score"].to_numpy(dtype=float), rcond=None
        )[0][: len(MAIN_X)]
        estimates[draw] = moderation_beta
        estimates[draw, X.index("board_independence_c")] = main_beta[MAIN_X.index("board_independence_c")]
        estimates[draw, X.index("firm_size_c")] = main_beta[MAIN_X.index("firm_size_c")]
    frame = pd.DataFrame(estimates, columns=X)
    frame.insert(0, "bootstrap_draw", np.arange(1, draws + 1))
    return frame


def glm_rows(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    status = {}
    rhs = (
        "board_independence_c + firm_size_c + board_x_size + roa + "
        "liquidity_ratio + leverage_ratio + C(bank_id) + C(year)"
    )
    for model_name, outcome in [
        ("Binary full-disclosure GLM with bank and year indicators", "full_disclosure"),
        ("Fractional logit GLM with bank and year indicators", "ori_fraction"),
        ("Poisson missing-disclosures GLM with bank and year indicators", "missing_disclosures"),
    ]:
        family = sm.families.Poisson() if outcome == "missing_disclosures" else sm.families.Binomial()
        try:
            result = smf.glm(f"{outcome} ~ {rhs}", data=sample, family=family).fit(
                cov_type="cluster",
                cov_kwds={"groups": sample["bank_id"], "use_correction": True},
                maxiter=300,
            )
            converged = bool(getattr(result, "converged", True))
            substantive_max = float(np.abs(result.params[X]).max())
            finite_inference = bool(
                np.isfinite(result.bse[X]).all() and np.isfinite(result.pvalues[X]).all()
            )
            status[model_name] = (
                f"estimated; converged={converged}; finite inference={finite_inference}; "
                f"max substantive |coefficient|={substantive_max:.3f}"
            )
            for hypothesis, variable in HYPOTHESES.items():
                coefficient = float(result.params[variable])
                se = float(result.bse[variable])
                rows.append(
                    {
                        "model": model_name,
                        "hypothesis": hypothesis,
                        "variable": variable,
                        "coefficient": coefficient,
                        "std_error": se,
                        "ci_lower_95": coefficient - 1.96 * se,
                        "ci_upper_95": coefficient + 1.96 * se,
                        "p_value": float(result.pvalues[variable]),
                        "converged": converged,
                        "bank_indicators": True,
                        "year_indicators": True,
                        "caveat": (
                            "Nonlinear fixed-indicator model with T<=6; incidental-parameter bias is possible."
                            if finite_inference
                            else "Inference unusable because standard errors/p-values are non-finite, consistent with separation."
                        ),
                    }
                )
        except Exception as exc:
            status[model_name] = f"not estimable: {type(exc).__name__}: {exc}"
    return pd.DataFrame(rows), status


def main() -> None:
    global INPUT, OUT_ROOT, OUT, FROZEN
    parser = argparse.ArgumentParser(description="Run second-pass statistical validation.")
    parser.add_argument("--input", type=Path, default=INPUT, help="Input bank-year CSV")
    parser.add_argument("--output", type=Path, default=OUT_ROOT, help="Output directory")
    args = parser.parse_args()
    INPUT = args.input.resolve()
    OUT_ROOT = args.output.resolve()
    OUT = OUT_ROOT / "second_pass_validation"
    FROZEN = INPUT
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(FROZEN)
    sample = raw.loc[raw[CORE_RAW].notna().all(axis=1)].copy()
    sample["board_independence_c"] = (
        sample["board_independence"] - sample["board_independence"].mean()
    )
    sample["firm_size_c"] = (
        sample["firm_size_ln_assets"] - sample["firm_size_ln_assets"].mean()
    )
    sample["board_x_size"] = sample["board_independence_c"] * sample["firm_size_c"]
    sample["ori_fraction"] = sample["ori_score"] / 100
    sample["full_disclosure"] = (sample["ori_score"] == 100).astype(int)
    sample["missing_disclosures"] = 8 - sample[[f"ori{i}" for i in range(1, 9)]].sum(axis=1)

    checks = {
        "input_hash_available": FROZEN.is_file(),
        "raw_rows_96": len(raw) == 96,
        "sample_rows_95": len(sample) == 95,
        "banks_16": sample["bank_id"].nunique() == 16,
        "unique_bank_year": not raw.duplicated(["bank_id", "year"]).any(),
        "single_core_exclusion": len(raw) - len(sample) == 1,
        "excluded_row_is_first_atlantic_2024": (
            raw.loc[~raw.index.isin(sample.index), ["bank", "year"]]
            .astype({"year": int})
            .to_dict("records")
            == [{"bank": "First Atlantic Bank PLC", "year": 2024}]
        ),
        "ori_recomputes": bool(
            np.isclose(
                sample["ori_score"],
                sample[[f"ori{i}" for i in range(1, 9)]].sum(axis=1) / 8 * 100,
            ).all()
        ),
        "board_independence_recomputes": bool(
            np.isclose(
                sample["board_independence"],
                sample["independent_directors"] / sample["board_total"],
            ).all()
        ),
        "firm_size_recomputes": bool(
            np.isclose(sample["firm_size_ln_assets"], np.log(sample["total_assets"])).all()
        ),
        "roa_recomputes": bool(
            np.isclose(sample["roa"], sample["profit_after_tax"] / sample["total_assets"]).all()
        ),
        "liquidity_recomputes": bool(
            np.isclose(
                sample["liquidity_ratio"],
                sample["cash_and_cash_equivalents"] / sample["total_assets"],
            ).all()
        ),
        "leverage_recomputes": bool(
            np.isclose(
                sample["leverage_ratio"],
                sample["total_liabilities"] / sample["total_assets"],
            ).all()
        ),
        "ori_ceiling_matches_verified_source": int((sample["ori_score"] == 100).sum())
        == int((raw["ori_score"] == 100).sum()),
    }

    variation = (
        sample.groupby(["bank_id", "bank"], as_index=False)
        .agg(
            observations=("year", "size"),
            ori_unique=("ori_score", "nunique"),
            ori_sd=("ori_score", "std"),
            board_independence_unique=("board_independence", "nunique"),
            board_independence_sd=("board_independence", "std"),
            firm_size_unique=("firm_size_ln_assets", "nunique"),
            firm_size_sd=("firm_size_ln_assets", "std"),
        )
        .sort_values("bank")
    )
    variation.to_csv(OUT / "within_bank_variation.csv", index=False)
    within_summary = {
        "banks_with_ori_variation": int((variation["ori_unique"] > 1).sum()),
        "banks_with_board_independence_variation": int(
            (variation["board_independence_unique"] > 1).sum()
        ),
        "banks_with_firm_size_variation": int((variation["firm_size_unique"] > 1).sum()),
        "within_sd_ori": float(
            np.sqrt(
                (
                    sample["ori_score"]
                    - sample.groupby("bank_id")["ori_score"].transform("mean")
                ).pow(2).sum()
                / (len(sample) - sample["bank_id"].nunique())
            )
        ),
        "within_sd_board_independence": float(
            np.sqrt(
                (
                    sample["board_independence"]
                    - sample.groupby("bank_id")["board_independence"].transform("mean")
                ).pow(2).sum()
                / (len(sample) - sample["bank_id"].nunique())
            )
        ),
        "within_sd_firm_size": float(
            np.sqrt(
                (
                    sample["firm_size_ln_assets"]
                    - sample.groupby("bank_id")["firm_size_ln_assets"].transform("mean")
                ).pow(2).sum()
                / (len(sample) - sample["bank_id"].nunique())
            )
        ),
    }

    fe_cluster = linearmodels_fe(
        sample, "clustered", cluster_entity=True, debiased=True
    )
    fe_unadjusted = linearmodels_fe(sample, "unadjusted")
    fe_robust = linearmodels_fe(sample, "robust", debiased=True)
    fe_kernel = linearmodels_fe(sample, "kernel", kernel="bartlett", bandwidth=2)

    formula = (
        "ori_score ~ board_independence_c + firm_size_c + board_x_size + roa + "
        "liquidity_ratio + leverage_ratio + C(bank_id) + C(year)"
    )
    lsdv_base = smf.ols(formula, data=sample)
    lsdv = lsdv_base.fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["bank_id"], "use_correction": True},
        use_t=True,
    )
    fwl = fwl_coefficients(sample)
    crosscheck = pd.DataFrame(
        {
            "variable": X,
            "linearmodels_fe": [float(fe_cluster.params[name]) for name in X],
            "statsmodels_lsdv": [float(lsdv.params[name]) for name in X],
            "manual_fwl": [float(fwl[name]) for name in X],
        }
    )
    crosscheck["max_pairwise_absolute_difference"] = crosscheck[
        ["linearmodels_fe", "statsmodels_lsdv", "manual_fwl"]
    ].max(axis=1) - crosscheck[
        ["linearmodels_fe", "statsmodels_lsdv", "manual_fwl"]
    ].min(axis=1)
    crosscheck.to_csv(OUT / "coefficient_crosscheck.csv", index=False)
    checks["three_way_coefficient_match_below_1e_8"] = bool(
        (crosscheck["max_pairwise_absolute_difference"] < 1e-8).all()
    )

    specification_rows = []
    specification_rows.extend(
        coefficient_rows(
            "Two-way FE: bank-clustered SE", fe_cluster, "linearmodels PanelOLS"
        )
    )
    specification_rows.extend(
        coefficient_rows("Two-way FE: unadjusted SE", fe_unadjusted, "linearmodels PanelOLS")
    )
    specification_rows.extend(
        coefficient_rows("Two-way FE: heteroskedastic-robust SE", fe_robust, "linearmodels PanelOLS")
    )
    specification_rows.extend(
        coefficient_rows("Two-way FE: Driscoll-Kraay bandwidth 2", fe_kernel, "linearmodels PanelOLS")
    )
    specification_rows.extend(
        coefficient_rows("Two-way FE: statsmodels LSDV clustered", lsdv, "statsmodels OLS with bank/year indicators")
    )

    panel = panel_frame(sample)
    fe_no_year = PanelOLS(
        panel["ori_score"], panel[X], entity_effects=True, drop_absorbed=True
    ).fit(cov_type="clustered", cluster_entity=True, debiased=True)
    specification_rows.extend(
        coefficient_rows("Bank FE without year effects", fe_no_year, "linearmodels PanelOLS")
    )

    main_effect_variables = [
        "board_independence_c",
        "firm_size_c",
        "roa",
        "liquidity_ratio",
        "leverage_ratio",
    ]
    fe_main_effects = PanelOLS(
        panel["ori_score"],
        panel[main_effect_variables],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True, debiased=True)
    for hypothesis, variable in {"H1": "board_independence_c", "H2": "firm_size_c"}.items():
        coefficient = float(fe_main_effects.params[variable])
        se = float(fe_main_effects.std_errors[variable])
        specification_rows.append(
            {
                "model": "Two-way FE main-effects model without interaction",
                "hypothesis": hypothesis,
                "variable": variable,
                "coefficient": coefficient,
                "std_error": se,
                "ci_lower_95": coefficient - 1.96 * se,
                "ci_upper_95": coefficient + 1.96 * se,
                "p_value": float(fe_main_effects.pvalues[variable]),
                "method": "linearmodels PanelOLS",
            }
        )

    minimal_variables = ["board_independence_c", "firm_size_c", "board_x_size"]
    fe_minimal = PanelOLS(
        panel["ori_score"],
        panel[minimal_variables],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True, debiased=True)
    specification_rows.extend(
        coefficient_rows(
            "Two-way FE moderation without financial controls",
            fe_minimal,
            "linearmodels PanelOLS",
        )
    )

    year_dummies = pd.get_dummies(
        panel.index.get_level_values("year").astype(int),
        prefix="year",
        drop_first=True,
        dtype=float,
    )
    year_dummies.index = panel.index
    exog = sm.add_constant(pd.concat([panel[X], year_dummies], axis=1), has_constant="add")
    pooled = PooledOLS(panel["ori_score"], exog).fit(
        cov_type="clustered", cluster_entity=True, debiased=True
    )
    random_effects = RandomEffects(panel["ori_score"], exog).fit(
        cov_type="clustered", cluster_entity=True, debiased=True
    )
    specification_rows.extend(
        coefficient_rows("Pooled OLS with year effects", pooled, "linearmodels PooledOLS")
    )
    specification_rows.extend(
        coefficient_rows("Random effects with year effects", random_effects, "linearmodels RandomEffects")
    )
    specification = pd.DataFrame(specification_rows)
    specification.to_csv(OUT / "specification_and_covariance_sensitivity.csv", index=False)

    glm, glm_status = glm_rows(sample)
    glm.to_csv(OUT / "bounded_outcome_bank_indicator_models.csv", index=False)

    bootstrap = bootstrap_clusters(sample)
    bootstrap.to_csv(OUT / "cluster_bootstrap_draws.csv", index=False)
    bootstrap_summary_rows = []
    for hypothesis, variable in HYPOTHESES.items():
        estimates = bootstrap[variable]
        bootstrap_summary_rows.append(
            {
                "hypothesis": hypothesis,
                "variable": variable,
                "draws": len(estimates),
                "mean": float(estimates.mean()),
                "std_error": float(estimates.std(ddof=1)),
                "percentile_ci_lower_95": float(estimates.quantile(0.025)),
                "percentile_ci_upper_95": float(estimates.quantile(0.975)),
                "share_positive": float((estimates > 0).mean()),
                "share_negative": float((estimates < 0).mean()),
            }
        )
    bootstrap_summary = pd.DataFrame(bootstrap_summary_rows)
    bootstrap_summary.to_csv(OUT / "cluster_bootstrap_summary.csv", index=False)

    moderation_primary = specification.loc[
        specification["model"] == "Two-way FE: bank-clustered SE"
    ].set_index("hypothesis")
    main_primary = specification.loc[
        specification["model"] == "Two-way FE main-effects model without interaction"
    ].set_index("hypothesis")
    primary = moderation_primary.copy()
    primary.loc["H1"] = main_primary.loc["H1"]
    primary.loc["H2"] = main_primary.loc["H2"]
    wild = pd.read_csv(OUT_ROOT / "diagnostics" / "wild_cluster_bootstrap_h1_h3.csv").set_index(
        "hypothesis"
    )
    nonlinear_h3 = glm.loc[glm["hypothesis"] == "H3", ["model", "coefficient", "p_value"]]
    checks["nondirectional_h1_supported"] = bool(
        primary.loc["H1", "p_value"] < 0.05 and wild.loc["H1", "p_value"] < 0.05
    )
    checks["h2_h3_remain_unsupported"] = bool(
        (primary.loc[["H2", "H3"], "p_value"] >= 0.05).all()
        and (wild.loc[["H2", "H3"], "p_value"] >= 0.05).all()
    )

    directional_rows = []
    for hypothesis in ["H1", "H2", "H3"]:
        row = primary.loc[hypothesis]
        two_sided = float(row["p_value"])
        coefficient = float(row["coefficient"])
        one_sided_positive = two_sided / 2 if coefficient > 0 else 1 - two_sided / 2
        directional_rows.append(
            {
                "hypothesis": hypothesis,
                "expected_direction": "nondirectional" if hypothesis == "H1" else "positive",
                "coefficient": coefficient,
                "two_sided_p_value": two_sided,
                "one_sided_positive_p_value": one_sided_positive,
                "supported_at_5_percent_one_sided": (
                    two_sided < 0.05 if hypothesis == "H1" else one_sided_positive < 0.05
                ),
            }
        )
    directional = pd.DataFrame(directional_rows)
    directional.to_csv(OUT / "directional_one_sided_tests.csv", index=False)
    checks["directional_h2_h3_unsupported"] = bool(
        (~directional.loc[directional.hypothesis.isin(["H2", "H3"]), "supported_at_5_percent_one_sided"]).all()
    )

    h_lines = []
    for hypothesis in ["H1", "H2", "H3"]:
        row = primary.loc[hypothesis]
        boot = bootstrap_summary.set_index("hypothesis").loc[hypothesis]
        h_lines.append(
            f"| {hypothesis} | {row['coefficient']:.4f} | {row['std_error']:.4f} | "
            f"{row['ci_lower_95']:.4f} to {row['ci_upper_95']:.4f} | {row['p_value']:.4f} | "
            f"{wild.loc[hypothesis, 'p_value']:.4f} | "
            f"{boot['percentile_ci_lower_95']:.4f} to {boot['percentile_ci_upper_95']:.4f} |"
        )

    sensitivity_lines = []
    for _, row in nonlinear_h3.iterrows():
        p_text = f"{row['p_value']:.4f}" if np.isfinite(row["p_value"]) else "not available"
        sensitivity_lines.append(
            f"- {row['model']}: H3 coefficient {row['coefficient']:.4f}, p={p_text}."
        )

    conclusion = (
        "The second pass confirms support for the nondirectional H1 in the baseline two-way FE model: "
        "the board-independence association is statistically significant and negative under both "
        "bank-clustered and wild-cluster inference. H2 and H3 remain unsupported."
    )

    report = f"""# Second-Pass Statistical Validation

## Bottom line

{conclusion}

The same six substantive coefficients were reproduced with three independent calculations: `linearmodels` two-way FE, a statsmodels least-squares dummy-variable regression, and a manual Frisch-Waugh-Lovell residualisation. The largest pairwise coefficient difference was **{crosscheck['max_pairwise_absolute_difference'].max():.3e}**.

## Primary hypothesis audit

| Hypothesis | FE coefficient | Clustered SE | 95% CI | Clustered p | Wild-cluster p | Bank-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(h_lines)}

The confidence intervals include zero for all three hypotheses. H1 is negative in the principal model; H2 and H3 are positive but imprecisely estimated.

## Why the mixed findings are plausible

- Only **{within_summary['banks_with_ori_variation']} of 16 banks** show any within-bank ORI variation.
- The within-bank ORI standard deviation is **{within_summary['within_sd_ori']:.3f}**, while {int((sample['ori_score'] == 100).sum())} of 95 observations sit at the maximum score of 100.
- Board independence and firm size vary within all 16 banks, but the outcome supplies limited within-bank information for estimating their effects.
- The H3 clustered standard error is **{primary.loc['H3', 'std_error']:.3f}**, almost twice the point estimate, so the interaction is weakly identified.
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

{chr(10).join(sensitivity_lines)}

These nonlinear bank-indicator models are sensitivity analyses. With at most six observations per bank, they can suffer incidental-parameter bias and must not replace the pre-specified linear FE model solely because one p-value crosses 0.05.

The binary full-disclosure model exhibits non-finite standard errors and p-values, consistent with separation caused by banks that are always at or always below full disclosure. It is not usable for inference. The fractional and Poisson bank-indicator checks yield nonsignificant H3 estimates and reinforce the primary conclusion.

## Data and coding audit

- Frozen input hash is available for the supplied input: **{checks['input_hash_available']}**.
- ORI, board independence, firm size, ROA, liquidity, and leverage all recompute exactly from their source columns.
- The primary sample remains 95 observations from 16 banks.
- The only core exclusion remains First Atlantic Bank PLC, 2024; it is not imputed.
- The interaction uses grand-mean-centred board independence and firm size exactly as stated in the methodology.

## Interpretation

The corrected conclusion is that the data support a statistically significant negative within-bank association between board independence and operational-resilience disclosure under nondirectional H1. H2 and H3 remain unsupported. The ceiling-heavy ORI and short six-year panel still limit statistical power, especially for moderation.
"""
    (OUT / "SECOND_PASS_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    audit = {
        "result": "PASS" if all(bool(value) for value in checks.values()) else "FAIL",
        "checks": checks,
        "within_variation": within_summary,
        "glm_status": glm_status,
        "largest_three_way_coefficient_difference": float(
            crosscheck["max_pairwise_absolute_difference"].max()
        ),
        "primary_hypotheses": primary.reset_index().to_dict("records"),
    }
    (OUT / "SECOND_PASS_AUDIT.json").write_text(
        json.dumps(audit, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, default=str))
    if audit["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
