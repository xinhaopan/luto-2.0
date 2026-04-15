from types import SimpleNamespace

import numpy as np

from luto.solvers import input_data


def test_strip_biodiversity_price_is_noop_when_price_zero(monkeypatch):
    data = SimpleNamespace(
        get_biodiversity_price_by_yr_idx=lambda yr_idx: 0.0,
    )
    ag_r_mrj = np.array([[[46.0]]], dtype=np.float32)

    def should_not_run(_data):
        raise AssertionError("ag biodiversity matrix should not be requested when bio price is zero")

    monkeypatch.setattr(
        input_data.ag_biodiversity,
        "get_bio_quality_score_mrj",
        should_not_run,
    )

    result = input_data._strip_biodiversity_price_from_ag_revenue(data, 0, ag_r_mrj)

    np.testing.assert_allclose(result, ag_r_mrj)


def test_non_ag_revenue_counts_biodiversity_once(monkeypatch):
    data = SimpleNamespace(
        YR_CAL_BASE=2010,
        lumaps={2010: np.array([15], dtype=np.int32)},
        get_biodiversity_price_by_year=lambda yr: 5.0,
        get_biodiversity_price_by_yr_idx=lambda yr_idx: 5.0,
    )

    ag_b_mrj = np.array([[[7.0]]], dtype=np.float32)
    ag_r_mrj = np.array([[[46.0]]], dtype=np.float32)  # 11 economic + 35 biodiversity
    non_ag_b_rk = np.array([[3.0]], dtype=np.float32)
    captured = {}

    def fake_get_rev_matrix(data_arg, target_year, ag_r_mrj_arg, lumap_arg):
        captured["ag_r_mrj"] = ag_r_mrj_arg.copy()
        return np.array([[2.0]], dtype=np.float32)

    monkeypatch.setattr(
        input_data.ag_biodiversity,
        "get_bio_quality_score_mrj",
        lambda data_arg: ag_b_mrj,
    )
    monkeypatch.setattr(
        input_data.non_ag_biodiversity,
        "get_breq_matrix",
        lambda data_arg, ag_b_arg, lumap_arg: non_ag_b_rk,
    )
    monkeypatch.setattr(
        input_data.non_ag_revenue,
        "get_rev_matrix",
        fake_get_rev_matrix,
    )

    result = input_data.get_non_ag_r_rk(data, ag_r_mrj, 2010, 2010)

    np.testing.assert_allclose(captured["ag_r_mrj"], np.array([[[11.0]]], dtype=np.float32))
    np.testing.assert_allclose(result, np.array([[17.0]], dtype=np.float32))


def test_non_ag_revenue_matches_upstream_when_bio_price_zero(monkeypatch):
    data = SimpleNamespace(
        YR_CAL_BASE=2010,
        lumaps={2010: np.array([15], dtype=np.int32)},
        get_biodiversity_price_by_year=lambda yr: 0.0,
        get_biodiversity_price_by_yr_idx=lambda yr_idx: 0.0,
    )
    ag_r_mrj = np.array([[[46.0]]], dtype=np.float32)
    captured = {}

    def fake_get_rev_matrix(data_arg, target_year, ag_r_mrj_arg, lumap_arg):
        captured["ag_r_mrj"] = ag_r_mrj_arg.copy()
        return np.array([[2.0]], dtype=np.float32)

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("non-ag biodiversity matrix should not be requested when bio price is zero")

    monkeypatch.setattr(
        input_data.non_ag_revenue,
        "get_rev_matrix",
        fake_get_rev_matrix,
    )
    monkeypatch.setattr(
        input_data.non_ag_biodiversity,
        "get_breq_matrix",
        should_not_run,
    )

    result = input_data.get_non_ag_r_rk(data, ag_r_mrj, 2010, 2010)

    np.testing.assert_allclose(captured["ag_r_mrj"], ag_r_mrj)
    np.testing.assert_allclose(result, np.array([[2.0]], dtype=np.float32))


def test_ag_management_revenue_counts_biodiversity_once(monkeypatch):
    data = SimpleNamespace(
        get_biodiversity_price_by_yr_idx=lambda yr_idx: 5.0,
    )

    ag_b_mrj = np.array([[[7.0]]], dtype=np.float32)
    ag_r_mrj = np.array([[[46.0]]], dtype=np.float32)  # 11 economic + 35 biodiversity
    ag_man_b_mrj = {"Test AM": np.array([[[3.0]]], dtype=np.float32)}
    captured = {}

    def fake_get_ag_mgt_revenue(data_arg, ag_r_mrj_arg, target_index):
        captured["ag_r_mrj"] = ag_r_mrj_arg.copy()
        return {"Test AM": np.array([[[2.0]]], dtype=np.float32)}

    monkeypatch.setattr(
        input_data.ag_biodiversity,
        "get_bio_quality_score_mrj",
        lambda data_arg: ag_b_mrj,
    )
    monkeypatch.setattr(
        input_data.ag_biodiversity,
        "get_ag_mgt_biodiversity_matrices",
        lambda data_arg, ag_b_arg, target_index: ag_man_b_mrj,
    )
    monkeypatch.setattr(
        input_data.ag_revenue,
        "get_agricultural_management_revenue_matrices",
        fake_get_ag_mgt_revenue,
    )

    result = input_data.get_ag_man_r_mrj(data, 0, ag_r_mrj)

    np.testing.assert_allclose(captured["ag_r_mrj"], np.array([[[11.0]]], dtype=np.float32))
    np.testing.assert_allclose(result["Test AM"], np.array([[[17.0]]], dtype=np.float32))


def test_ag_management_revenue_matches_upstream_when_bio_price_zero(monkeypatch):
    data = SimpleNamespace(
        get_biodiversity_price_by_yr_idx=lambda yr_idx: 0.0,
    )
    ag_r_mrj = np.array([[[46.0]]], dtype=np.float32)
    captured = {}

    def fake_get_ag_mgt_revenue(data_arg, ag_r_mrj_arg, target_index):
        captured["ag_r_mrj"] = ag_r_mrj_arg.copy()
        return {"Test AM": np.array([[[2.0]]], dtype=np.float32)}

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("ag-management biodiversity matrix should not be requested when bio price is zero")

    monkeypatch.setattr(
        input_data.ag_revenue,
        "get_agricultural_management_revenue_matrices",
        fake_get_ag_mgt_revenue,
    )
    monkeypatch.setattr(
        input_data.ag_biodiversity,
        "get_ag_mgt_biodiversity_matrices",
        should_not_run,
    )

    result = input_data.get_ag_man_r_mrj(data, 0, ag_r_mrj)

    np.testing.assert_allclose(captured["ag_r_mrj"], ag_r_mrj)
    np.testing.assert_allclose(result["Test AM"], np.array([[[2.0]]], dtype=np.float32))
