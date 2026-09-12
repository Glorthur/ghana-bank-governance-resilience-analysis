import numpy as np
import pandas as pd

from conftest import FROZEN


def test_frozen_panel_shape_and_keys():
    frame = pd.read_csv(FROZEN)
    assert len(frame) == 96
    assert frame["bank"].nunique() == 16
    assert set(frame["year"]) == set(range(2020, 2026))
    assert not frame.duplicated(["bank", "year"]).any()


def test_panel_derived_fields_and_bounds():
    frame = pd.read_csv(FROZEN)
    assert frame["ori_score"].dropna().between(0, 100).all()
    assert frame["ori_score"].isna().sum() == 1
    assert frame["board_independence"].dropna().between(0, 1).all()
    assert frame["board_independence"].isna().sum() == 1
    assert (frame["total_assets"] > 0).all()
    np.testing.assert_allclose(frame["firm_size_ln_assets"], np.log(frame["total_assets"]), rtol=0, atol=1e-8)
    np.testing.assert_allclose(frame["roa"], frame["profit_after_tax"] / frame["total_assets"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(frame["liquidity_ratio"], frame["cash_and_cash_equivalents"] / frame["total_assets"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(frame["leverage_ratio"], frame["total_liabilities"] / frame["total_assets"], rtol=0, atol=1e-8)

