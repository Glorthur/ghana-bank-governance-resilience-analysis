import numpy as np
import pandas as pd

from conftest import FROZEN, PANEL


def test_frozen_panel_shape_and_keys():
    frame = pd.read_csv(FROZEN)
    assert len(frame) == 96
    assert frame["bank"].nunique() == 16
    assert set(frame["year"]) == set(range(2020, 2026))
    assert not frame.duplicated(["bank", "year"]).any()


def test_panel_derived_fields_and_bounds():
    frame = pd.read_csv(FROZEN)
    assert frame["ori_score"].between(0, 100).all()
    assert frame["board_independence"].between(0, 1).all()
    assert (frame["total_assets"] > 0).all()
    np.testing.assert_allclose(frame["firm_size_ln_assets"], np.log(frame["total_assets"]), rtol=0, atol=1e-8)
    np.testing.assert_allclose(frame["roa"], frame["net_income"] / frame["total_assets"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(frame["liquidity_ratio"], frame["liquid_assets"] / frame["total_assets"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(frame["leverage_ratio"], frame["total_liabilities"] / frame["total_assets"], rtol=0, atol=1e-8)


def test_analysis_panel_has_expected_sample_flags():
    frame = pd.read_csv(PANEL)
    assert len(frame) == 96
    assert frame["core_sample"].sum() == 95
    assert frame.loc[frame["core_sample"], "bank"].nunique() == 16
    assert frame["car_available"].sum() == 70
    assert frame.loc[frame["car_available"], "bank"].nunique() == 15
    assert frame["npl_available"].sum() == 43
    assert frame.loc[frame["npl_available"], "bank"].nunique() == 10

