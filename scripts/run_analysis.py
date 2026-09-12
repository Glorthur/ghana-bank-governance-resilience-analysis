from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
import warnings
import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from wildboottest.wildboottest import wildboottest


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "frozen" / "bank_year_master.csv"
OUTPUT_ROOT = ROOT / "outputs"
FROZEN = OUTPUT_ROOT / "input_frozen" / "bank_year_master.csv"
DATA_DIR = OUTPUT_ROOT / "data"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
DIAGNOSTIC_DIR = OUTPUT_ROOT / "diagnostics"
LOG_DIR = OUTPUT_ROOT / "logs"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"

CORE = [
    "ori_score",
    "board_independence",
    "firm_size_ln_assets",
    "roa",
    "liquidity_ratio",
    "leverage_ratio",
]
PREDICTORS = [
    "board_independence_c",
    "firm_size_c",
    "board_x_size",
    "roa",
    "liquidity_ratio",
    "leverage_ratio",
]
LABELS = {
    "board_independence_c": "Board independence (centred)",
    "firm_size_c": "Firm size (centred ln assets)",
    "board_x_size": "Board independence x firm size",
    "roa": "Return on assets",
    "liquidity_ratio": "Liquidity ratio",
    "leverage_ratio": "Leverage ratio",
    "capital_adequacy_ratio": "Capital adequacy ratio",
}
SEED = 20260902


def ensure_dirs() -> None:
    for directory in [
        FROZEN.parent,
        DATA_DIR,
        TABLE_DIR,
        FIGURE_DIR,
        DIAGNOSTIC_DIR,
        LOG_DIR,
        MANIFEST_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def stars(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def clean_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)
    return " ".join(text.split())[:1000]


def excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.columns:
        if pd.api.types.is_string_dtype(safe[column].dtype) or safe[column].dtype == object:
            safe[column] = safe[column].map(
                lambda value: re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", value)
                if isinstance(value, str)
                else value
            )
    return safe


def write_markdown_table(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.write_text(frame.to_markdown(index=index) + "\n", encoding="utf-8")


def freeze_input() -> dict[str, str]:
    if not INPUT.exists():
        raise FileNotFoundError(f"Input master not found: {INPUT}")
    source_hash = sha256(INPUT)
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUT, FROZEN)
    frozen_hash = sha256(FROZEN)
    if source_hash != frozen_hash:
        raise RuntimeError("Frozen input hash does not match the verified source master.")
    manifest = {
        "source_path": str(INPUT.relative_to(ROOT)) if INPUT.is_relative_to(ROOT) else INPUT.name,
        "frozen_path": str(FROZEN.relative_to(OUTPUT_ROOT)),
        "sha256": frozen_hash,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (MANIFEST_DIR / "INPUT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def validate_and_prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "bank_id",
        "bank",
        "year",
        "ori_score",
        "board_independence",
        "firm_size_ln_assets",
        "roa",
        "liquidity_ratio",
        "leverage_ratio",
        "capital_adequacy_ratio",
        "npl_ratio",
        "total_assets",
        "total_liabilities",
        "cash_and_cash_equivalents",
        "profit_after_tax",
        "board_total",
        "independent_directors",
    }
    missing_columns = sorted(required.difference(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = raw.copy()
    numeric = sorted(required.difference({"bank_id", "bank"}))
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    checks: dict[str, object] = {}
    checks["rows_96"] = len(df) == 96
    checks["banks_16"] = df["bank_id"].nunique() == 16
    checks["years_2020_2025"] = sorted(df["year"].dropna().unique().tolist()) == list(
        range(2020, 2026)
    )
    checks["unique_bank_year"] = not df.duplicated(["bank_id", "year"]).any()
    checks["ori_range"] = bool(df["ori_score"].dropna().between(0, 100).all())
    checks["board_independence_range"] = bool(df["board_independence"].dropna().between(0, 1).all())
    checks["positive_assets"] = bool((df["total_assets"].dropna() > 0).all())
    checks["board_counts_valid"] = bool(
        (df["independent_directors"].dropna() >= 0).all()
        and (
            df.loc[df["independent_directors"].notna(), "independent_directors"]
            <= df.loc[df["independent_directors"].notna(), "board_total"]
        ).all()
    )

    tolerances = {
        "firm_size_formula": np.isclose(
            df["firm_size_ln_assets"], np.log(df["total_assets"]), rtol=1e-10, atol=1e-10
        )
        | df["firm_size_ln_assets"].isna(),
        "roa_formula": np.isclose(
            df["roa"], df["profit_after_tax"] / df["total_assets"], rtol=1e-10, atol=1e-10
        )
        | df["roa"].isna(),
        "liquidity_formula": np.isclose(
            df["liquidity_ratio"],
            df["cash_and_cash_equivalents"] / df["total_assets"],
            rtol=1e-10,
            atol=1e-10,
        )
        | df["liquidity_ratio"].isna(),
        "leverage_formula": np.isclose(
            df["leverage_ratio"],
            df["total_liabilities"] / df["total_assets"],
            rtol=1e-10,
            atol=1e-10,
        )
        | df["leverage_ratio"].isna(),
        "board_independence_formula": np.isclose(
            df["board_independence"],
            df["independent_directors"] / df["board_total"],
            rtol=1e-10,
            atol=1e-10,
        )
        | df["board_independence"].isna(),
    }
    for name, mask in tolerances.items():
        checks[name] = bool(mask.all())

    df["core_sample"] = df[CORE].notna().all(axis=1)
    df["car_sample"] = df[CORE + ["capital_adequacy_ratio"]].notna().all(axis=1)
    df["npl_sample"] = df[
        [
            "npl_ratio",
            "board_independence",
            "firm_size_ln_assets",
            "roa",
            "liquidity_ratio",
            "leverage_ratio",
        ]
    ].notna().all(axis=1)
    core = df.loc[df["core_sample"]]
    means = {
        "board_independence_mean": float(core["board_independence"].mean()),
        "firm_size_mean": float(core["firm_size_ln_assets"].mean()),
    }
    df["board_independence_c"] = (
        df["board_independence"] - means["board_independence_mean"]
    )
    df["firm_size_c"] = df["firm_size_ln_assets"] - means["firm_size_mean"]
    df["board_x_size"] = df["board_independence_c"] * df["firm_size_c"]
    df["ori_fraction"] = df["ori_score"] / 100.0
    df["full_disclosure"] = np.where(
        df["ori_score"].notna(), (df["ori_score"] == 100).astype(int), np.nan
    )

    checks.update(
        {
            "core_rows_95": int(df["core_sample"].sum()) == 95,
            "core_banks_16": df.loc[df["core_sample"], "bank_id"].nunique() == 16,
            "car_rows": int(df["car_sample"].sum()),
            "car_banks": int(df.loc[df["car_sample"], "bank_id"].nunique()),
            "npl_rows": int(df["npl_sample"].sum()),
            "npl_banks": int(df.loc[df["npl_sample"], "bank_id"].nunique()),
            "ori_at_100": int((df.loc[df["core_sample"], "ori_score"] == 100).sum()),
        }
    )
    boolean_checks = [value for value in checks.values() if isinstance(value, (bool, np.bool_))]
    if not all(boolean_checks):
        failures = [key for key, value in checks.items() if isinstance(value, (bool, np.bool_)) and not value]
        raise ValueError(f"Validation failed: {failures}")
    return df, {"checks": checks, "centering_means": means}


def build_sample_register(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "bank_id",
        "bank",
        "year",
        "core_sample",
        "car_sample",
        "npl_sample",
        "ori_score",
        "capital_adequacy_ratio",
        "npl_ratio",
        "row_validation_status",
        "source_file",
        "source_sha256",
    ]
    register = df[columns].copy()
    register["core_exclusion_reason"] = np.where(
        register["core_sample"], "included", "missing ORI and/or core governance variable"
    )
    register["car_exclusion_reason"] = np.where(
        register["car_sample"], "included", "missing CAR and/or core variable"
    )
    register["npl_exclusion_reason"] = np.where(
        register["npl_sample"], "included", "missing NPL and/or required covariate"
    )
    return register.sort_values(["bank", "year"])


def descriptive_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    core = df.loc[df["core_sample"]].copy()
    variables = [
        "ori_score",
        "board_independence",
        "firm_size_ln_assets",
        "roa",
        "liquidity_ratio",
        "leverage_ratio",
    ]
    desc = core[variables].describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index()
    desc = desc.rename(columns={"index": "variable", "50%": "median"})
    desc = desc[
        ["variable", "count", "mean", "std", "min", "25%", "median", "75%", "max"]
    ]

    extended = df[
        ["capital_adequacy_ratio", "npl_ratio"]
    ].describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index()
    extended = extended.rename(columns={"index": "variable", "50%": "median"})
    extended = extended[
        ["variable", "count", "mean", "std", "min", "25%", "median", "75%", "max"]
    ]

    by_year = (
        core.groupby("year")
        .agg(
            observations=("ori_score", "size"),
            banks=("bank_id", "nunique"),
            ori_mean=("ori_score", "mean"),
            ori_sd=("ori_score", "std"),
            ori_median=("ori_score", "median"),
            full_disclosure_rate=("full_disclosure", "mean"),
            board_independence_mean=("board_independence", "mean"),
            firm_size_mean=("firm_size_ln_assets", "mean"),
        )
        .reset_index()
    )
    by_bank = (
        core.groupby("bank", as_index=False)
        .agg(
            observations=("ori_score", "size"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            ori_mean=("ori_score", "mean"),
            ori_sd=("ori_score", "std"),
            full_disclosure_rate=("full_disclosure", "mean"),
            board_independence_mean=("board_independence", "mean"),
            firm_size_mean=("firm_size_ln_assets", "mean"),
        )
        .sort_values("bank")
    )

    ori_distribution = (
        core["ori_score"].value_counts().sort_index().rename_axis("ori_score").reset_index(name="count")
    )
    ori_distribution["percent"] = 100 * ori_distribution["count"] / len(core)

    ori_columns = [f"ori{i}" for i in range(1, 9)]
    component = pd.DataFrame(
        {
            "ori_item": ori_columns,
            "disclosures": [int(core[column].sum()) for column in ori_columns],
            "available": [int(core[column].notna().sum()) for column in ori_columns],
            "prevalence_percent": [100 * core[column].mean() for column in ori_columns],
        }
    )
    return {
        "descriptive_core": desc,
        "descriptive_extended": extended,
        "descriptive_by_year": by_year,
        "descriptive_by_bank": by_bank,
        "ori_distribution": ori_distribution,
        "ori_component_prevalence": component,
    }


def correlation_and_vif(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    core = df.loc[df["core_sample"]].copy()
    columns = ["ori_score"] + PREDICTORS
    corr = core[columns].corr(method="pearson")
    x = sm.add_constant(core[PREDICTORS], has_constant="add")
    vif = pd.DataFrame(
        {
            "variable": x.columns,
            "vif": [variance_inflation_factor(x.values, i) for i in range(x.shape[1])],
        }
    )
    return corr, vif


def prepare_panel(sample: pd.DataFrame) -> pd.DataFrame:
    return sample.sort_values(["bank_id", "year"]).set_index(["bank_id", "year"])


def year_dummies(sample: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(sample["year"].astype(int), prefix="year", drop_first=True, dtype=float)


def fit_panel_models(df: pd.DataFrame) -> dict[str, object]:
    sample = df.loc[df["core_sample"]].copy()
    panel = prepare_panel(sample)
    y = panel["ori_score"]
    yd = pd.get_dummies(
        panel.index.get_level_values("year").astype(int), prefix="year", drop_first=True, dtype=float
    )
    yd.index = panel.index

    main_vars = [
        "board_independence_c",
        "firm_size_c",
        "roa",
        "liquidity_ratio",
        "leverage_ratio",
    ]
    mod_vars = PREDICTORS
    x_main = sm.add_constant(pd.concat([panel[main_vars], yd], axis=1), has_constant="add")
    x_mod = sm.add_constant(pd.concat([panel[mod_vars], yd], axis=1), has_constant="add")

    models: dict[str, object] = {}
    models["pooled_main"] = PooledOLS(y, x_main).fit(
        cov_type="clustered", cluster_entity=True, debiased=True
    )
    models["fe_main"] = PanelOLS(
        y,
        panel[main_vars],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True, debiased=True)
    models["re_main"] = RandomEffects(y, x_main).fit(
        cov_type="clustered", cluster_entity=True, debiased=True
    )
    models["pooled_moderation"] = PooledOLS(y, x_mod).fit(
        cov_type="clustered", cluster_entity=True, debiased=True
    )
    models["fe_moderation"] = PanelOLS(
        y,
        panel[mod_vars],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True, debiased=True)
    models["re_moderation"] = RandomEffects(y, x_mod).fit(
        cov_type="clustered", cluster_entity=True, debiased=True
    )

    models["fe_moderation_unadjusted"] = PanelOLS(
        y,
        panel[mod_vars],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(cov_type="unadjusted")
    models["re_moderation_unadjusted"] = RandomEffects(y, x_mod).fit(cov_type="unadjusted")

    main_formula = (
        "ori_score ~ board_independence_c + firm_size_c + roa + "
        "liquidity_ratio + leverage_ratio + C(bank_id) + C(year)"
    )
    models["dummy_fe_main_ols_model"] = smf.ols(main_formula, data=sample)
    models["dummy_fe_main_ols"] = models["dummy_fe_main_ols_model"].fit(
        cov_type="cluster", cov_kwds={"groups": sample["bank_id"], "use_correction": True}
    )
    formula = (
        "ori_score ~ board_independence_c + firm_size_c + board_x_size + roa + "
        "liquidity_ratio + leverage_ratio + C(bank_id) + C(year)"
    )
    models["dummy_fe_ols_model"] = smf.ols(formula, data=sample)
    models["dummy_fe_ols"] = models["dummy_fe_ols_model"].fit(
        cov_type="cluster", cov_kwds={"groups": sample["bank_id"], "use_correction": True}
    )
    return models


def model_table(models: dict[str, object]) -> pd.DataFrame:
    order = [
        "pooled_main",
        "fe_main",
        "re_main",
        "pooled_moderation",
        "fe_moderation",
        "re_moderation",
    ]
    rows: list[dict[str, object]] = []
    for name in order:
        result = models[name]
        for variable in [
            "board_independence_c",
            "firm_size_c",
            "board_x_size",
            "roa",
            "liquidity_ratio",
            "leverage_ratio",
        ]:
            if variable not in result.params.index:
                continue
            rows.append(
                {
                    "model": name,
                    "variable": variable,
                    "label": LABELS[variable],
                    "coefficient": float(result.params[variable]),
                    "std_error": float(result.std_errors[variable]),
                    "statistic": float(result.tstats[variable]),
                    "p_value": float(result.pvalues[variable]),
                    "significance": stars(float(result.pvalues[variable])),
                    "observations": int(result.nobs),
                    "banks": 16,
                    "bank_effects": name.startswith("fe_"),
                    "year_effects": True,
                    "standard_errors": "clustered by bank",
                    "r_squared": float(result.rsquared),
                }
            )
    return pd.DataFrame(rows)


def hausman_test(models: dict[str, object]) -> pd.DataFrame:
    fe = models["fe_moderation_unadjusted"]
    re = models["re_moderation_unadjusted"]
    common = [name for name in PREDICTORS if name in fe.params.index and name in re.params.index]
    diff = fe.params[common] - re.params[common]
    cov_diff = fe.cov.loc[common, common] - re.cov.loc[common, common]
    statistic = float(diff.T @ np.linalg.pinv(cov_diff.to_numpy()) @ diff)
    statistic = max(statistic, 0.0)
    df_test = len(common)
    pvalue = float(stats.chi2.sf(statistic, df_test))
    return pd.DataFrame(
        [
            {
                "test": "Conventional Hausman FE versus RE",
                "statistic": statistic,
                "degrees_of_freedom": df_test,
                "p_value": pvalue,
                "interpretation": (
                    "Reject RE consistency in favour of FE" if pvalue < 0.05 else "Do not reject RE consistency"
                ),
                "caveat": "Conventional covariance-difference test; interpret with the Mundlak check.",
            }
        ]
    )


def mundlak_test(df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    sample = df.loc[df["core_sample"]].copy()
    for column in PREDICTORS:
        sample[f"mean_{column}"] = sample.groupby("bank_id")[column].transform("mean")
    panel = prepare_panel(sample)
    y = panel["ori_score"]
    yd = pd.get_dummies(
        panel.index.get_level_values("year").astype(int), prefix="year", drop_first=True, dtype=float
    )
    yd.index = panel.index
    mean_columns = [f"mean_{column}" for column in PREDICTORS]
    x = sm.add_constant(pd.concat([panel[PREDICTORS + mean_columns], yd], axis=1), has_constant="add")
    result = RandomEffects(y, x).fit(
        cov_type="clustered", cluster_entity=True, debiased=True
    )
    restrictions = np.zeros((len(mean_columns), len(result.params)))
    for row, variable in enumerate(mean_columns):
        restrictions[row, result.params.index.get_loc(variable)] = 1.0
    test = result.wald_test(restrictions)
    table = pd.DataFrame(
        [
            {
                "test": "Mundlak joint test of bank-level regressor means",
                "statistic": float(np.asarray(test.stat).squeeze()),
                "degrees_of_freedom": len(mean_columns),
                "p_value": float(test.pval),
                "interpretation": (
                    "Bank effects are correlated with regressors; FE is preferred"
                    if test.pval < 0.05
                    else "No joint evidence against the RE orthogonality assumption"
                ),
            }
        ]
    )
    return table, result


def simple_slopes(models: dict[str, object], df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = models["fe_moderation"]
    sample = df.loc[df["core_sample"]]
    size_sd = float(sample["firm_size_c"].std(ddof=1))
    values = [-size_sd, 0.0, size_sd]
    labels = ["Firm size: -1 SD", "Firm size: mean", "Firm size: +1 SD"]
    covariance = result.cov
    rows = []
    for label, value in zip(labels, values):
        slope = float(result.params["board_independence_c"] + value * result.params["board_x_size"])
        variance = float(
            covariance.loc["board_independence_c", "board_independence_c"]
            + value**2 * covariance.loc["board_x_size", "board_x_size"]
            + 2 * value * covariance.loc["board_independence_c", "board_x_size"]
        )
        se = math.sqrt(max(variance, 0.0))
        statistic = slope / se if se else np.nan
        pvalue = 2 * stats.t.sf(abs(statistic), df=max(int(result.df_resid), 1)) if se else np.nan
        rows.append(
            {
                "firm_size_level": label,
                "firm_size_c": value,
                "board_independence_slope": slope,
                "std_error": se,
                "ci_lower_95": slope - 1.96 * se,
                "ci_upper_95": slope + 1.96 * se,
                "p_value": pvalue,
                "significance": stars(pvalue),
            }
        )
    slopes = pd.DataFrame(rows)

    grid = np.linspace(sample["firm_size_c"].quantile(0.05), sample["firm_size_c"].quantile(0.95), 100)
    grid_rows = []
    for value in grid:
        slope = float(result.params["board_independence_c"] + value * result.params["board_x_size"])
        variance = float(
            covariance.loc["board_independence_c", "board_independence_c"]
            + value**2 * covariance.loc["board_x_size", "board_x_size"]
            + 2 * value * covariance.loc["board_independence_c", "board_x_size"]
        )
        se = math.sqrt(max(variance, 0.0))
        grid_rows.append(
            {
                "firm_size_c": value,
                "firm_size_ln_assets": value + sample["firm_size_ln_assets"].mean(),
                "marginal_effect": slope,
                "ci_lower_95": slope - 1.96 * se,
                "ci_upper_95": slope + 1.96 * se,
            }
        )
    return slopes, pd.DataFrame(grid_rows)


def wild_cluster_tests(models: dict[str, object], sample: pd.DataFrame) -> pd.DataFrame:
    cluster_codes = pd.factorize(sample["bank_id"], sort=True)[0].astype(np.int64)
    rows = []
    for hypothesis, parameter, model_key in [
        ("H1", "board_independence_c", "dummy_fe_main_ols_model"),
        ("H2", "firm_size_c", "dummy_fe_main_ols_model"),
        ("H3", "board_x_size", "dummy_fe_ols_model"),
    ]:
        try:
            result = wildboottest(
                models[model_key],
                B=9999,
                cluster=cluster_codes,
                param=parameter,
                weights_type="rademacher",
                seed=SEED,
                adj=True,
                cluster_adj=True,
                parallel=False,
                show=False,
            )
            row = result.reset_index().iloc[0]
            p_candidates = [column for column in result.columns if "p-value" in str(column).lower() or "pvalue" in str(column).lower()]
            t_candidates = [
                column
                for column in result.columns
                if "t-value" in str(column).lower()
                or "tstat" in str(column).lower()
                or str(column).lower() == "statistic"
            ]
            pvalue = float(row[p_candidates[0]]) if p_candidates else float(result.iloc[0, -1])
            statistic = float(row[t_candidates[0]]) if t_candidates else np.nan
            rows.append(
                {
                    "hypothesis": hypothesis,
                    "parameter": parameter,
                    "bootstrap_draws": 9999,
                    "t_statistic": statistic,
                    "p_value": pvalue,
                    "status": "estimated",
                }
            )
        except Exception as exc:  # pragma: no cover - recorded as an output
            rows.append(
                {
                    "hypothesis": hypothesis,
                    "parameter": parameter,
                    "bootstrap_draws": 9999,
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "status": f"not estimable: {clean_error(exc)}",
                }
            )
    return pd.DataFrame(rows)


def wooldridge_serial_test(df: pd.DataFrame) -> pd.DataFrame:
    sample = df.loc[df["core_sample"]].sort_values(["bank_id", "year"]).copy()
    diff_columns = ["ori_score"] + PREDICTORS
    differences = sample.groupby("bank_id")[diff_columns].diff()
    first_stage = pd.concat([sample[["bank_id", "year"]], differences.add_prefix("d_")], axis=1).dropna()
    y = first_stage["d_ori_score"]
    x = first_stage[[f"d_{column}" for column in PREDICTORS]]
    residuals = sm.OLS(y, x).fit().resid
    residual_frame = first_stage[["bank_id", "year"]].copy()
    residual_frame["residual"] = residuals
    residual_frame["lag_residual"] = residual_frame.groupby("bank_id")["residual"].shift(1)
    second = residual_frame.dropna()
    result = sm.OLS(second["residual"], second[["lag_residual"]]).fit(
        cov_type="cluster", cov_kwds={"groups": second["bank_id"], "use_correction": True}
    )
    coefficient = float(result.params["lag_residual"])
    se = float(result.bse["lag_residual"])
    statistic = (coefficient + 0.5) / se
    pvalue = float(2 * stats.t.sf(abs(statistic), df=max(second["bank_id"].nunique() - 1, 1)))
    return pd.DataFrame(
        [
            {
                "test": "Wooldridge first-difference serial-correlation test",
                "null_hypothesis": "First-difference residual coefficient equals -0.5 (no AR(1) in levels)",
                "estimated_coefficient": coefficient,
                "std_error": se,
                "statistic": statistic,
                "p_value": pvalue,
                "interpretation": (
                    "Evidence of first-order serial correlation" if pvalue < 0.05 else "No evidence of first-order serial correlation"
                ),
            }
        ]
    )


def pesaran_cd(residuals: pd.Series) -> pd.DataFrame:
    frame = residuals.rename("residual").reset_index()
    wide = frame.pivot(index="year", columns="bank_id", values="residual")
    banks = list(wide.columns)
    weighted_sum = 0.0
    pairs = 0
    pair_rows = []
    for i, first in enumerate(banks):
        for second in banks[i + 1 :]:
            pair = wide[[first, second]].dropna()
            if len(pair) < 3 or pair[first].std() == 0 or pair[second].std() == 0:
                continue
            correlation = float(pair[first].corr(pair[second]))
            weighted_sum += math.sqrt(len(pair)) * correlation
            pairs += 1
            pair_rows.append(correlation)
    statistic = math.sqrt(2 / (len(banks) * (len(banks) - 1))) * weighted_sum
    pvalue = float(2 * stats.norm.sf(abs(statistic)))
    return pd.DataFrame(
        [
            {
                "test": "Pesaran CD test on FE residuals",
                "banks": len(banks),
                "usable_pairs": pairs,
                "mean_pairwise_correlation": float(np.mean(pair_rows)),
                "statistic": statistic,
                "p_value": pvalue,
                "interpretation": (
                    "Evidence of cross-sectional dependence" if pvalue < 0.05 else "No evidence of cross-sectional dependence"
                ),
            }
        ]
    )


def diagnostic_tests(models: dict[str, object], df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sample = df.loc[df["core_sample"]].copy()
    dummy_result = models["dummy_fe_ols"]
    bp_stat, bp_p, f_stat, f_p = het_breuschpagan(
        dummy_result.resid, dummy_result.model.exog
    )
    hetero = pd.DataFrame(
        [
            {
                "test": "Breusch-Pagan heteroskedasticity test",
                "lm_statistic": bp_stat,
                "lm_p_value": bp_p,
                "f_statistic": f_stat,
                "f_p_value": f_p,
                "interpretation": (
                    "Evidence of heteroskedasticity" if bp_p < 0.05 else "No evidence of heteroskedasticity"
                ),
            }
        ]
    )

    influence = models["dummy_fe_ols_model"].fit().get_influence()
    influence_frame = sample[["bank", "bank_id", "year", "ori_score"]].copy()
    influence_frame["fitted"] = models["dummy_fe_ols_model"].fit().fittedvalues
    influence_frame["residual"] = models["dummy_fe_ols_model"].fit().resid
    influence_frame["leverage"] = influence.hat_matrix_diag
    influence_frame["cooks_distance"] = influence.cooks_distance[0]
    influence_frame["studentized_residual"] = influence.resid_studentized_external
    influence_frame["high_cooks_distance"] = influence_frame["cooks_distance"] > 4 / len(sample)
    influence_frame = influence_frame.sort_values("cooks_distance", ascending=False)

    fe_residuals = models["fe_moderation"].resids
    return {
        "heteroskedasticity": hetero,
        "serial_correlation": wooldridge_serial_test(df),
        "cross_sectional_dependence": pesaran_cd(fe_residuals),
        "influence": influence_frame,
    }


def fit_fe_for_sample(sample: pd.DataFrame, dependent: str, extra: list[str] | None = None):
    extra = extra or []
    panel = prepare_panel(sample)
    variables = PREDICTORS + extra
    return PanelOLS(
        panel[dependent],
        panel[variables],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True, debiased=True)


def robustness_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, object]] = []
    status: dict[str, str] = {}

    car_sample = df.loc[df["car_sample"]].copy()
    try:
        car_result = fit_fe_for_sample(
            car_sample, "ori_score", extra=["capital_adequacy_ratio"]
        )
        for variable in PREDICTORS + ["capital_adequacy_ratio"]:
            if variable in car_result.params.index:
                rows.append(
                    {
                        "model": "CAR-augmented two-way FE",
                        "outcome": "ori_score",
                        "variable": variable,
                        "coefficient": float(car_result.params[variable]),
                        "std_error": float(car_result.std_errors[variable]),
                        "p_value": float(car_result.pvalues[variable]),
                        "observations": int(car_result.nobs),
                        "banks": car_sample["bank_id"].nunique(),
                        "bank_effects": True,
                        "year_effects": True,
                        "standard_errors": "clustered by bank",
                    }
                )
        status["car_model"] = "estimated"
    except Exception as exc:
        status["car_model"] = f"not estimable: {clean_error(exc)}"

    core = df.loc[df["core_sample"]].copy()
    formula_rhs = (
        "board_independence_c + firm_size_c + board_x_size + roa + "
        "liquidity_ratio + leverage_ratio + C(year)"
    )
    for model_name, outcome in [
        ("Full-disclosure logit GLM", "full_disclosure"),
        ("Fractional-response logit GLM", "ori_fraction"),
    ]:
        try:
            result = smf.glm(
                f"{outcome} ~ {formula_rhs}", data=core, family=sm.families.Binomial()
            ).fit(cov_type="cluster", cov_kwds={"groups": core["bank_id"], "use_correction": True})
            for variable in PREDICTORS:
                rows.append(
                    {
                        "model": model_name,
                        "outcome": outcome,
                        "variable": variable,
                        "coefficient": float(result.params[variable]),
                        "std_error": float(result.bse[variable]),
                        "p_value": float(result.pvalues[variable]),
                        "observations": int(result.nobs),
                        "banks": core["bank_id"].nunique(),
                        "bank_effects": False,
                        "year_effects": True,
                        "standard_errors": "clustered by bank",
                    }
                )
            status[model_name] = "estimated"
        except Exception as exc:
            status[model_name] = f"not estimable: {clean_error(exc)}"

    npl_sample = df.loc[df["npl_sample"]].copy()
    try:
        npl_result = fit_fe_for_sample(npl_sample, "npl_ratio")
        for variable in PREDICTORS:
            if variable in npl_result.params.index:
                rows.append(
                    {
                        "model": "NPL separate-outcome two-way FE",
                        "outcome": "npl_ratio",
                        "variable": variable,
                        "coefficient": float(npl_result.params[variable]),
                        "std_error": float(npl_result.std_errors[variable]),
                        "p_value": float(npl_result.pvalues[variable]),
                        "observations": int(npl_result.nobs),
                        "banks": npl_sample["bank_id"].nunique(),
                        "bank_effects": True,
                        "year_effects": True,
                        "standard_errors": "clustered by bank",
                    }
                )
        status["npl_model"] = "estimated; separate outcome, not an ORI proxy"
    except Exception as exc:
        status["npl_model"] = f"not estimable: {clean_error(exc)}"

    leave_rows = []
    for bank_id, bank_name in core[["bank_id", "bank"]].drop_duplicates().itertuples(index=False):
        reduced = core.loc[core["bank_id"] != bank_id]
        try:
            result = fit_fe_for_sample(reduced, "ori_score")
            leave_rows.append(
                {
                    "omitted_bank_id": bank_id,
                    "omitted_bank": bank_name,
                    "observations": int(result.nobs),
                    "banks": reduced["bank_id"].nunique(),
                    "board_independence_coefficient": float(result.params["board_independence_c"]),
                    "board_independence_p_value": float(result.pvalues["board_independence_c"]),
                    "interaction_coefficient": float(result.params["board_x_size"]),
                    "interaction_p_value": float(result.pvalues["board_x_size"]),
                    "status": "estimated",
                }
            )
        except Exception as exc:
            leave_rows.append(
                {
                    "omitted_bank_id": bank_id,
                    "omitted_bank": bank_name,
                    "observations": len(reduced),
                    "banks": reduced["bank_id"].nunique(),
                    "board_independence_coefficient": np.nan,
                    "board_independence_p_value": np.nan,
                    "interaction_coefficient": np.nan,
                    "interaction_p_value": np.nan,
                    "status": f"not estimable: {clean_error(exc)}",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(leave_rows), status


def make_figures(
    df: pd.DataFrame,
    descriptive: dict[str, pd.DataFrame],
    marginal_grid: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    core = df.loc[df["core_sample"]].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    distribution = descriptive["ori_distribution"]
    bars = ax.bar(distribution["ori_score"].astype(str), distribution["count"], color=["#D55E00", "#E69F00", "#0072B2"])
    ax.bar_label(bars, labels=[f"{count} ({pct:.1f}%)" for count, pct in zip(distribution["count"], distribution["percent"])], padding=3)
    ax.set(title="Operational Resilience Index distribution, 2020-2025", xlabel="ORI score", ylabel="Bank-year observations")
    ax.set_ylim(0, max(distribution["count"]) * 1.18)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_01_ori_distribution.png", dpi=220)
    plt.close(fig)

    by_year = descriptive["descriptive_by_year"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(by_year["year"], by_year["ori_mean"], marker="o", color="#0072B2", linewidth=2)
    ax.fill_between(
        by_year["year"],
        by_year["ori_mean"] - by_year["ori_sd"],
        np.minimum(100, by_year["ori_mean"] + by_year["ori_sd"]),
        color="#56B4E9",
        alpha=0.2,
        label="Mean +/- 1 SD",
    )
    ax.set(title="Mean Operational Resilience Index by year", xlabel="Year", ylabel="Mean ORI score", ylim=(65, 102))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_02_ori_year_trend.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(marginal_grid["firm_size_ln_assets"], marginal_grid["marginal_effect"], color="#0072B2", linewidth=2)
    ax.fill_between(
        marginal_grid["firm_size_ln_assets"],
        marginal_grid["ci_lower_95"],
        marginal_grid["ci_upper_95"],
        color="#56B4E9",
        alpha=0.25,
        label="95% confidence interval",
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set(
        title="Marginal effect of board independence across firm size",
        xlabel="Firm size (natural log of total assets)",
        ylabel="Marginal effect on ORI",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_03_marginal_effect_board_independence.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    plot = leave_one_out.sort_values("interaction_coefficient")
    ax.errorbar(
        plot["interaction_coefficient"],
        np.arange(len(plot)),
        fmt="o",
        color="#009E73",
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(np.arange(len(plot)), labels=plot["omitted_bank"])
    ax.set(title="Leave-one-bank-out interaction estimates", xlabel="Board independence x firm size coefficient", ylabel="Omitted bank")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_04_leave_one_bank_out_interaction.png", dpi=220)
    plt.close(fig)


def hypothesis_decision(coefficient: float, pvalue: float, expected: str = "positive") -> str:
    direction_ok = expected == "nondirectional" or (coefficient > 0 if expected == "positive" else coefficient < 0)
    if pvalue < 0.05 and direction_ok:
        return "supported at the 5% level"
    if pvalue < 0.10 and direction_ok:
        return "weakly supported at the 10% level"
    if pvalue < 0.05 and not direction_ok:
        return "statistically significant in the opposite direction"
    return "not supported statistically"


def write_interpretation(
    df: pd.DataFrame,
    models: dict[str, object],
    hausman: pd.DataFrame,
    mundlak: pd.DataFrame,
    simple_slopes_frame: pd.DataFrame,
    wild: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    robustness: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    robustness_status: dict[str, str],
) -> None:
    fe_main = models["fe_main"]
    fe = models["fe_moderation"]
    b1, p1 = float(fe_main.params["board_independence_c"]), float(fe_main.pvalues["board_independence_c"])
    b2, p2 = float(fe_main.params["firm_size_c"]), float(fe_main.pvalues["firm_size_c"])
    b3, p3 = float(fe.params["board_x_size"]), float(fe.pvalues["board_x_size"])
    wild_lookup = wild.set_index("hypothesis")["p_value"].to_dict()
    leave_sign = float((np.sign(leave_one_out["interaction_coefficient"]) == np.sign(b3)).mean())
    leave_sig = int((leave_one_out["interaction_p_value"] < 0.05).sum())
    pooled = models["pooled_moderation"]
    pooled_b1 = float(pooled.params["board_independence_c"])
    pooled_p1 = float(pooled.pvalues["board_independence_c"])
    pooled_b3 = float(pooled.params["board_x_size"])
    pooled_p3 = float(pooled.pvalues["board_x_size"])
    robust_lookup = robustness.set_index(["model", "variable"])
    frac_b3 = float(robust_lookup.loc[("Fractional-response logit GLM", "board_x_size"), "coefficient"])
    frac_p3 = float(robust_lookup.loc[("Fractional-response logit GLM", "board_x_size"), "p_value"])
    binary_b3 = float(robust_lookup.loc[("Full-disclosure logit GLM", "board_x_size"), "coefficient"])
    binary_p3 = float(robust_lookup.loc[("Full-disclosure logit GLM", "board_x_size"), "p_value"])
    car_b3 = float(robust_lookup.loc[("CAR-augmented two-way FE", "board_x_size"), "coefficient"])
    car_p3 = float(robust_lookup.loc[("CAR-augmented two-way FE", "board_x_size"), "p_value"])
    h1 = hypothesis_decision(b1, p1, expected="nondirectional")
    h2 = hypothesis_decision(b2, p2)
    h3 = hypothesis_decision(b3, p3)
    core = df.loc[df["core_sample"]]

    memo = f"""# Analysis Interpretation Memo

## Executive finding

The staged fixed-effects analysis uses **{int(fe.nobs)} bank-year observations from 16 banks**, with bank and year effects and bank-clustered standard errors. H1 and H2 are evaluated in the baseline direct-effects model, while H3 is evaluated in the moderation model. The outcome is highly ceiling-concentrated: **{int((core['ori_score'] == 100).sum())} of {len(core)} observations ({100 * (core['ori_score'] == 100).mean():.1f}%) score 100**. Results therefore describe within-bank changes in a disclosure index and must not be framed as causal evidence of realised operational resilience.

The baseline FE model produces a statistically significant negative board-independence coefficient of **{b1:.4f} (p={p1:.4f})**. Under the nondirectional H1, this supports the existence of a within-bank association, but its negative sign must be reported and explained. The moderation model's interaction coefficient is **{b3:.4f} (p={p3:.4f})** and is not statistically significant.

## Hypothesis results

| Hypothesis | FE coefficient | Clustered p-value | Wild-cluster p-value | Decision |
|---|---:|---:|---:|---|
| H1: Board independence is associated with ORI | {b1:.4f} | {p1:.4f} | {fmt(wild_lookup.get('H1', np.nan))} | {h1} |
| H2: Firm size positively affects ORI | {b2:.4f} | {p2:.4f} | {fmt(wild_lookup.get('H2', np.nan))} | {h2} |
| H3: Board independence x firm size -> ORI | {b3:.4f} | {p3:.4f} | {fmt(wild_lookup.get('H3', np.nan))} | {h3} |

H1 and H2 are direct effects from the baseline FE model. The H3 interaction and the simple slopes below come from the moderation FE model.

## Moderation interpretation

The interaction estimate is **{b3:.4f}**. Its sign is retained in **{100 * leave_sign:.1f}%** of the 16 leave-one-bank-out specifications, but it is significant at 5% in **{leave_sig} of 16**. The simple-slope results are:

{simple_slopes_frame.to_markdown(index=False)}

Use `figures/figure_03_marginal_effect_board_independence.png` when discussing H3. Emphasise the confidence interval and avoid claiming moderation where it overlaps zero.

## Model selection

- Hausman p-value: **{hausman.loc[0, 'p_value']:.4f}**. {hausman.loc[0, 'interpretation']}.
- Mundlak p-value: **{mundlak.loc[0, 'p_value']:.4f}**. {mundlak.loc[0, 'interpretation']}.
- Fixed effects remain the principal model because the research question concerns within-bank change and unobserved time-invariant bank characteristics are substantively plausible.

## Diagnostics

- Heteroskedasticity test p-value: **{diagnostics['heteroskedasticity'].loc[0, 'lm_p_value']:.4f}**.
- Wooldridge serial-correlation p-value: **{diagnostics['serial_correlation'].loc[0, 'p_value']:.4f}**.
- Pesaran CD p-value: **{diagnostics['cross_sectional_dependence'].loc[0, 'p_value']:.4f}**.
- Bank-clustered inference is used throughout the linear panel models. Wild-cluster bootstrap p-values are supplied for H1-H3 because there are only 16 clusters.

## Robustness boundaries

- CAR model: {robustness_status.get('car_model', 'not run')}. It uses a reduced sample and is not co-primary.
- Full-disclosure sensitivity: {robustness_status.get('Full-disclosure logit GLM', 'not run')}.
- Fractional-response sensitivity: {robustness_status.get('Fractional-response logit GLM', 'not run')}.
- NPL model: {robustness_status.get('npl_model', 'not run')}. NPL is a separate outcome and must not be described as an ORI proxy.

The CAR-augmented FE interaction is **{car_b3:.4f} (p={car_p3:.4f})**. The year-adjusted pooled binary full-disclosure interaction is **{binary_b3:.4f} (p={binary_p3:.4f})**, and the year-adjusted pooled fractional-response interaction is **{frac_b3:.4f} (p={frac_p3:.4f})**. The fractional specification is significant, but it does not include bank fixed effects and conflicts with the principal FE result and wild-cluster inference. It is therefore sensitivity evidence, not sufficient support for H3.

## Required caveats

1. The panel is small: 16 banks over at most six years.
2. ORI takes the observed non-missing values {', '.join(f'{value:g}' for value in sorted(core['ori_score'].dropna().unique()))}, with a strong ceiling effect.
3. The annual-report index measures disclosure evidence, not directly observed disruption recovery performance.
4. CAR and NPL results rely on substantially smaller, non-randomly available samples.
5. Associations are not causal estimates.
6. Do not add bank age unless a complete, independently verified bank-year measure is assembled.
"""
    (OUTPUT_ROOT / "ANALYSIS_INTERPRETATION_MEMO.md").write_text(memo, encoding="utf-8")

    outline = f"""# Chapter Four Drafting Outline

## 4.1 Introduction

State that the chapter reports the verified 2020-2025 bank panel, descriptive evidence, panel-model diagnostics, hypothesis tests, and bounded-outcome/reduced-sample robustness checks.

## 4.2 Sample and data quality

- Intended panel: 96 bank-years from 16 banks.
- Primary complete-case sample: {int(fe.nobs)} bank-years.
- Explain the single excluded observation using `tables/analysis_sample_register.csv`.
- Report CAR and NPL availability separately.
- State that ORI equals 100 in {int((core['ori_score'] == 100).sum())} observations and discuss the ceiling limitation before modelling.

Recommended exhibits: sample-flow table, missingness table, and ORI distribution figure.

## 4.3 Descriptive statistics

Use `tables/table_01_descriptive_core.csv`, `table_03_descriptive_by_year.csv`, and `table_06_ori_component_prevalence.csv`. Describe magnitude and variation without interpreting correlations as effects.

## 4.4 Correlations and preliminary diagnostics

Use the correlation matrix, VIF table, and diagnostics in `diagnostics/`. Explain why centred predictors are used in the moderation model.

## 4.5 Model choice

Present pooled OLS, fixed effects, and random effects side by side. Report both Hausman and Mundlak results. Identify two-way fixed effects with bank-clustered standard errors as the principal specification.

## 4.6 Hypothesis tests

### 4.6.1 Board independence and ORI (H1)

Report the baseline direct-effects FE coefficient {b1:.4f}, clustered p-value {p1:.4f}, and wild-cluster p-value {fmt(wild_lookup.get('H1', np.nan))}. Conclude: **{h1}**. State explicitly that the estimated association is negative.

### 4.6.2 Firm size and ORI (H2)

Report the baseline direct-effects FE coefficient {b2:.4f}, clustered p-value {p2:.4f}, and wild-cluster p-value {fmt(wild_lookup.get('H2', np.nan))}. Conclude: **{h2}**.

### 4.6.3 Moderating effect of firm size (H3)

Report the interaction coefficient {b3:.4f}, clustered p-value {p3:.4f}, and wild-cluster p-value {fmt(wild_lookup.get('H3', np.nan))}. Conclude: **{h3}**. Follow with the simple-slopes table and marginal-effects figure.

## 4.7 Robustness checks

Present CAR, full-disclosure, fractional-response, leave-one-bank-out, and NPL analyses as sensitivity evidence. Keep the 95-row FE model visibly primary and label every reduced sample.

## 4.8 Discussion

Relate only statistically supported patterns to agency, resource-dependence, and contingency theory. Discuss unsupported hypotheses directly. Balance governance benefits against information asymmetry, competence constraints, larger-bank complexity, cyberattack surface, legacy systems, and bureaucratic friction.

## 4.9 Chapter conclusion

Summarise the evidence without causal language. Carry the ceiling effect, small panel, and disclosure-versus-realised-resilience distinction into Chapter Five.
"""
    (OUTPUT_ROOT / "CHAPTER_FOUR_DRAFTING_OUTLINE.md").write_text(outline, encoding="utf-8")


def export_results(
    df: pd.DataFrame,
    validation: dict,
    sample_register: pd.DataFrame,
    descriptive: dict[str, pd.DataFrame],
    corr: pd.DataFrame,
    vif: pd.DataFrame,
    regression: pd.DataFrame,
    hausman: pd.DataFrame,
    mundlak: pd.DataFrame,
    slopes: pd.DataFrame,
    marginal_grid: pd.DataFrame,
    wild: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    robustness: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    robustness_status: dict[str, str],
) -> None:
    df.to_csv(DATA_DIR / "analysis_ready_panel.csv", index=False)
    df.loc[df["core_sample"]].to_csv(DATA_DIR / "primary_model_sample_95.csv", index=False)
    sample_register.to_csv(TABLE_DIR / "analysis_sample_register.csv", index=False)

    table_names = {
        "descriptive_core": "table_01_descriptive_core",
        "descriptive_extended": "table_02_descriptive_extended",
        "descriptive_by_year": "table_03_descriptive_by_year",
        "descriptive_by_bank": "table_04_descriptive_by_bank",
        "ori_distribution": "table_05_ori_distribution",
        "ori_component_prevalence": "table_06_ori_component_prevalence",
    }
    for key, frame in descriptive.items():
        stem = table_names[key]
        frame.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
        write_markdown_table(frame.round(4), TABLE_DIR / f"{stem}.md")

    corr.to_csv(TABLE_DIR / "table_07_correlation_matrix.csv")
    write_markdown_table(corr.round(4).reset_index(), TABLE_DIR / "table_07_correlation_matrix.md")
    vif.to_csv(TABLE_DIR / "table_08_vif.csv", index=False)
    regression.to_csv(TABLE_DIR / "table_09_panel_regressions.csv", index=False)
    write_markdown_table(regression.round(4), TABLE_DIR / "table_09_panel_regressions.md")
    slopes.to_csv(TABLE_DIR / "table_10_simple_slopes.csv", index=False)
    write_markdown_table(slopes.round(4), TABLE_DIR / "table_10_simple_slopes.md")
    robustness.to_csv(TABLE_DIR / "table_11_robustness_models.csv", index=False)
    write_markdown_table(robustness.round(4), TABLE_DIR / "table_11_robustness_models.md")
    leave_one_out.to_csv(TABLE_DIR / "table_12_leave_one_bank_out.csv", index=False)
    marginal_grid.to_csv(DATA_DIR / "marginal_effect_grid.csv", index=False)

    hausman.to_csv(DIAGNOSTIC_DIR / "hausman_test.csv", index=False)
    mundlak.to_csv(DIAGNOSTIC_DIR / "mundlak_test.csv", index=False)
    wild.to_csv(DIAGNOSTIC_DIR / "wild_cluster_bootstrap_h1_h3.csv", index=False)
    for name, frame in diagnostics.items():
        frame.to_csv(DIAGNOSTIC_DIR / f"{name}.csv", index=False)

    missingness = pd.DataFrame(
        {
            "variable": df.columns,
            "nonmissing": [int(df[column].notna().sum()) for column in df.columns],
            "missing": [int(df[column].isna().sum()) for column in df.columns],
            "missing_percent": [100 * df[column].isna().mean() for column in df.columns],
        }
    )
    missingness.to_csv(TABLE_DIR / "missingness_report.csv", index=False)

    validation_payload = {
        **validation,
        "robustness_status": robustness_status,
        "python": sys.version,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "statsmodels": sm.__version__,
        },
    }
    (MANIFEST_DIR / "VALIDATION_REPORT.json").write_text(
        json.dumps(validation_payload, indent=2, default=str), encoding="utf-8"
    )
    report_lines = [
        "# Data Validation Report",
        "",
        "## Overall assessment: Ready for analysis with stated caveats",
        "",
        f"- Intended rows: {len(df)}",
        f"- Banks: {df['bank_id'].nunique()}",
        f"- Primary complete observations: {int(df['core_sample'].sum())}",
        f"- CAR observations: {int(df['car_sample'].sum())}",
        f"- NPL observations: {int(df['npl_sample'].sum())}",
        f"- ORI observations at 100: {int((df.loc[df['core_sample'], 'ori_score'] == 100).sum())}",
        "",
        "All identity, range, uniqueness, and recomputed formula checks passed.",
        "",
        "The single primary-sample exclusion is First Atlantic Bank PLC, 2024, because the available source is a financial-only summary without the ORI and governance evidence required for the core model. It is not imputed.",
        "",
        "All CAR and NPL exclusions are documented bank-year by bank-year in `tables/analysis_sample_register.csv`.",
    ]
    (OUTPUT_ROOT / "DATA_VALIDATION_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    with pd.ExcelWriter(OUTPUT_ROOT / "ANALYSIS_TABLES.xlsx", engine="openpyxl") as writer:
        excel_safe(descriptive["descriptive_core"]).to_excel(writer, sheet_name="Descriptives", index=False)
        excel_safe(descriptive["descriptive_by_year"]).to_excel(writer, sheet_name="By Year", index=False)
        excel_safe(descriptive["ori_component_prevalence"]).to_excel(writer, sheet_name="ORI Components", index=False)
        excel_safe(corr).to_excel(writer, sheet_name="Correlations")
        excel_safe(vif).to_excel(writer, sheet_name="VIF", index=False)
        excel_safe(regression).to_excel(writer, sheet_name="Panel Models", index=False)
        excel_safe(slopes).to_excel(writer, sheet_name="Simple Slopes", index=False)
        excel_safe(robustness).to_excel(writer, sheet_name="Robustness", index=False)
        excel_safe(leave_one_out).to_excel(writer, sheet_name="Leave One Out", index=False)
        excel_safe(hausman).to_excel(writer, sheet_name="Hausman", index=False)
        excel_safe(mundlak).to_excel(writer, sheet_name="Mundlak", index=False)
        excel_safe(wild).to_excel(writer, sheet_name="Wild Bootstrap", index=False)


def write_output_manifest() -> None:
    files = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() not in {".xlsx", ".xlsm"} and path.name not in {
            "OUTPUT_MANIFEST.json",
            "FINAL_ANALYSIS_AUDIT.json",
        }:
            files.append(
                {
                    "path": str(path.relative_to(OUTPUT_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (MANIFEST_DIR / "OUTPUT_MANIFEST.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> None:
    global INPUT, OUTPUT_ROOT, FROZEN, DATA_DIR, TABLE_DIR, FIGURE_DIR, DIAGNOSTIC_DIR, LOG_DIR, MANIFEST_DIR
    parser = argparse.ArgumentParser(description="Run the audited panel analysis.")
    parser.add_argument("--input", type=Path, default=INPUT, help="Input bank-year CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT, help="Output directory")
    args = parser.parse_args()
    INPUT = args.input.resolve()
    OUTPUT_ROOT = args.output.resolve()
    FROZEN = OUTPUT_ROOT / "input_frozen" / "bank_year_master.csv"
    DATA_DIR = OUTPUT_ROOT / "data"
    TABLE_DIR = OUTPUT_ROOT / "tables"
    FIGURE_DIR = OUTPUT_ROOT / "figures"
    DIAGNOSTIC_DIR = OUTPUT_ROOT / "diagnostics"
    LOG_DIR = OUTPUT_ROOT / "logs"
    MANIFEST_DIR = OUTPUT_ROOT / "manifests"
    warnings.filterwarnings("default")
    np.random.seed(SEED)
    ensure_dirs()
    input_manifest = freeze_input()
    raw = pd.read_csv(FROZEN)
    df, validation = validate_and_prepare(raw)
    validation["input_manifest"] = input_manifest
    register = build_sample_register(df)
    descriptive = descriptive_outputs(df)
    corr, vif = correlation_and_vif(df)
    models = fit_panel_models(df)
    regression = model_table(models)
    hausman = hausman_test(models)
    mundlak, _ = mundlak_test(df)
    slopes, marginal_grid = simple_slopes(models, df)
    core_sample = df.loc[df["core_sample"]].copy()
    wild = wild_cluster_tests(models, core_sample)
    diagnostics = diagnostic_tests(models, df)
    robustness, leave_one_out, robustness_status = robustness_outputs(df)
    make_figures(df, descriptive, marginal_grid, leave_one_out)
    export_results(
        df,
        validation,
        register,
        descriptive,
        corr,
        vif,
        regression,
        hausman,
        mundlak,
        slopes,
        marginal_grid,
        wild,
        diagnostics,
        robustness,
        leave_one_out,
        robustness_status,
    )
    write_interpretation(
        df,
        models,
        hausman,
        mundlak,
        slopes,
        wild,
        diagnostics,
        robustness,
        leave_one_out,
        robustness_status,
    )
    write_output_manifest()
    summary = {
        "status": "PASS",
        "primary_observations": int(df["core_sample"].sum()),
        "banks": int(df.loc[df["core_sample"], "bank_id"].nunique()),
        "car_observations": int(df["car_sample"].sum()),
        "npl_observations": int(df["npl_sample"].sum()),
        "output_root": (
            str(OUTPUT_ROOT.relative_to(ROOT))
            if OUTPUT_ROOT.is_relative_to(ROOT)
            else OUTPUT_ROOT.name
        ),
    }
    (LOG_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
