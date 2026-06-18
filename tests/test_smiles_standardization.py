from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.standardization import (
    add_activity_label,
    convert_to_molar,
    ic50_to_pic50,
    remove_invalid_activity,
    run_activity_standardization,
)


@pytest.mark.unit
class TestConvertToMolar:
    def test_nm(self):
        assert convert_to_molar(100, "nM") == pytest.approx(1e-7)

    def test_um(self):
        assert convert_to_molar(1, "uM") == pytest.approx(1e-6)

    def test_unicode_um(self):
        assert convert_to_molar(1, "µM") == pytest.approx(1e-6)

    def test_unknown_none(self):
        assert convert_to_molar(100, "xyz") is None

    def test_case_insensitive(self):
        assert convert_to_molar(100, "NM") == pytest.approx(1e-7)


@pytest.mark.unit
class TestIc50ToPic50:
    def test_1nm_is_9(self):
        assert ic50_to_pic50(1e-9) == pytest.approx(9.0, abs=1e-5)

    def test_1um_is_6(self):
        assert ic50_to_pic50(1e-6) == pytest.approx(6.0, abs=1e-5)

    def test_zero_none(self):
        assert ic50_to_pic50(0) is None

    def test_negative_none(self):
        assert ic50_to_pic50(-1e-9) is None

    def test_nan_none(self):
        assert ic50_to_pic50(float("nan")) is None


@pytest.mark.unit
class TestAddActivityLabel:
    def test_all_active(self):
        df = pd.DataFrame({"pic50": [8.0, 7.0, 6.0]})
        assert add_activity_label(df)["binary_label"].sum() == 3

    def test_all_inactive(self):
        df = pd.DataFrame({"pic50": [4.0, 3.5]})
        assert add_activity_label(df)["binary_label"].sum() == 0

    def test_three_classes(self):
        df = pd.DataFrame({"pic50": [8.0, 5.5, 3.0]})
        classes = set(add_activity_label(df)["activity_class"].astype(str))
        assert classes == {"active", "gray", "inactive"}


@pytest.mark.unit
class TestRemoveInvalidActivity:
    def test_removes_nan(self):
        df = pd.DataFrame({"pic50": [7.0, np.nan, 6.0]})
        assert len(remove_invalid_activity(df)) == 2

    def test_removes_out_of_range(self):
        df = pd.DataFrame({"pic50": [7.0, 15.0, -1.0]})
        assert len(remove_invalid_activity(df)) == 1


@pytest.mark.unit
class TestRunActivityStandardization:
    def test_runs(self, tiny_molecules_df):
        result = run_activity_standardization(tiny_molecules_df)
        assert "pic50" in result.columns
        assert result["pic50"].notna().all()

    def test_has_labels(self, tiny_molecules_df):
        assert "binary_label" in run_activity_standardization(tiny_molecules_df).columns

    def test_pic50_range(self, tiny_molecules_df):
        result = run_activity_standardization(tiny_molecules_df)
        assert (result["pic50"] >= 3.0).all()
        assert (result["pic50"] <= 12.0).all()
