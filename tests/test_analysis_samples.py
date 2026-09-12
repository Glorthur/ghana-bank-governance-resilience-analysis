import pandas as pd

from conftest import PANEL


def test_analysis_panel_has_expected_sample_flags():
    frame = pd.read_csv(PANEL)
    assert len(frame) == 96
    assert frame["core_sample"].sum() == 95
    assert frame.loc[frame["core_sample"], "bank"].nunique() == 16
    assert frame["car_sample"].sum() == 70
    assert frame.loc[frame["car_sample"], "bank"].nunique() == 15
    assert frame["npl_sample"].sum() == 43
    assert frame.loc[frame["npl_sample"], "bank"].nunique() == 10


def test_single_primary_exclusion_is_documented_bank_year():
    frame = pd.read_csv(PANEL)
    excluded = frame.loc[~frame["core_sample"], ["bank", "year"]]
    assert excluded.to_dict("records") == [
        {"bank": "First Atlantic Bank PLC", "year": 2024}
    ]
