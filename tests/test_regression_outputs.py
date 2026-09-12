import subprocess
import sys

import numpy as np
import pandas as pd

from conftest import ROOT


EXPECTED = {
    ("pooled_main", "board_independence_c"): -17.913331322403256,
    ("fe_main", "board_independence_c"): -18.76021517327689,
    ("re_main", "board_independence_c"): -17.927712440055988,
    ("pooled_moderation", "board_independence_c"): -16.605438224012083,
    ("fe_moderation", "board_independence_c"): -15.325417776157146,
    ("re_moderation", "board_independence_c"): -14.868904678405778,
}


def test_documented_cli_reproduces_regression_table(tmp_path):
    command = [sys.executable, "scripts/run_analysis.py", "--input", "data/frozen/bank_year_master.csv", "--output", str(tmp_path)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]
    produced = pd.read_csv(tmp_path / "tables" / "table_09_panel_regressions.csv")
    for key, expected in EXPECTED.items():
        row = produced.set_index(["model", "variable"]).loc[key]
        assert np.isclose(row["coefficient"], expected, atol=1e-8)
        assert row["observations"] == 95
        assert row["banks"] == 16
