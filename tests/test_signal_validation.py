"""Offline unit tests for python/upstream/signal_validation.py (pure pandas/numpy/scipy)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "upstream"))
import signal_validation as sv  # noqa: E402


def panel(values, n_dates=None, start="2020-01-03"):
    a = np.asarray(values, dtype=float)
    idx = pd.date_range(start, periods=a.shape[0], freq="W-FRI")
    return pd.DataFrame(a, index=idx, columns=[f"S{i:03d}" for i in range(a.shape[1])])


# ── rank_ic_series ───────────────────────────────────────────────────────────
def test_rank_ic_perfect_and_reversed():
    rng = np.random.default_rng(0)
    f = panel(rng.normal(size=(5, 40)))
    assert np.allclose(sv.rank_ic_series(f * 3 + 1, f), 1.0)
    assert np.allclose(sv.rank_ic_series(-f, f), -1.0)


def test_rank_ic_matches_scipy_spearman_with_ties_and_nans():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 6, size=(6, 45)).astype(float)         # many ties
    f = rng.normal(size=(6, 45))
    s[rng.random(s.shape) < 0.1] = np.nan
    f[rng.random(f.shape) < 0.1] = np.nan
    S, F = panel(s), panel(f)
    got = sv.rank_ic_series(S, F)
    for i in range(6):
        ok = ~np.isnan(s[i]) & ~np.isnan(f[i])
        want = stats.spearmanr(s[i][ok], f[i][ok])[0]
        assert got.iloc[i] == pytest.approx(want, abs=1e-12)


def test_rank_ic_ranks_only_names_valid_in_both():
    # S000 has a huge signal but no forward return; it must not shift anyone's rank.
    s = np.arange(40, dtype=float)[None, :].repeat(2, axis=0)
    f = s.copy()
    s[:, 0] = 1e9
    f[:, 0] = np.nan
    ic = sv.rank_ic_series(panel(s), panel(f))
    assert np.allclose(ic, 1.0)


def test_rank_ic_min_names_boundary_default_30():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(1, 40))
    s = f.copy()
    s[0, 30:] = np.nan                       # exactly 30 valid names
    assert sv.rank_ic_series(panel(s), panel(f)).iloc[0] == pytest.approx(1.0)
    s[0, 29] = np.nan                        # 29 valid names
    assert np.isnan(sv.rank_ic_series(panel(s), panel(f)).iloc[0])


def test_rank_ic_min_names_argument_and_constant_row():
    rng = np.random.default_rng(3)
    f = rng.normal(size=(1, 12))
    assert np.isnan(sv.rank_ic_series(panel(f), panel(f)).iloc[0])        # 12 < 30
    assert sv.rank_ic_series(panel(f), panel(f), min_names=12).iloc[0] == pytest.approx(1.0)
    assert np.isnan(sv.rank_ic_series(panel(f), panel(f), min_names=13).iloc[0])
    const = panel(np.ones((1, 40)))
    assert np.isnan(sv.rank_ic_series(const, panel(rng.normal(size=(1, 40)))).iloc[0])


def test_rank_ic_inner_aligns_index_and_columns():
    rng = np.random.default_rng(4)
    s = panel(rng.normal(size=(6, 45)))
    f = s.iloc[2:, :40]
    ic = sv.rank_ic_series(s, f)
    assert list(ic.index) == list(s.index[2:])
    assert np.allclose(ic, 1.0)


# ── newey_west_t ─────────────────────────────────────────────────────────────
def nw_oracle(x, lags):
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    acov = lambda k: float(np.sum(e[k:] * e[: n - k])) / n
    var = acov(0) + sum(2 * (1 - k / (lags + 1)) * acov(k) for k in range(1, min(lags, n - 1) + 1))
    t = x.mean() / math.sqrt(var / n)
    return x.mean(), t, 2 * stats.t.sf(abs(t), n - 1)


@pytest.mark.parametrize("lags", [0, 1, 2, 5])
def test_newey_west_matches_hand_formula(lags):
    rng = np.random.default_rng(5)
    x = 0.1 + np.cumsum(rng.normal(size=60)) * 0.05 + rng.normal(size=60)
    got = sv.newey_west_t(x, lags)
    assert got == pytest.approx(nw_oracle(x, lags), rel=1e-12)


def test_newey_west_lags_zero_is_plain_t_with_biased_variance():
    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 7.0])
    mu, t, p = sv.newey_west_t(x, 0)
    assert mu == pytest.approx(x.mean())
    assert t == pytest.approx(x.mean() / math.sqrt(x.var(ddof=0) / len(x)))
    assert p == pytest.approx(2 * stats.t.sf(abs(t), 5))


def test_newey_west_positive_autocorr_shrinks_t():
    x = np.repeat([0.3, -0.1, 0.4, -0.2, 0.5, -0.3, 0.2, 0.0], 4) + 0.2
    t0 = sv.newey_west_t(x, 0)[1]
    t4 = sv.newey_west_t(x, 4)[1]
    assert abs(t4) < abs(t0)


def test_newey_west_n_boundary_and_nan_handling():
    nan3 = (float("nan"),) * 3
    assert all(math.isnan(v) for v in sv.newey_west_t([1.0, 2.0], 1))
    assert all(math.isnan(v) for v in sv.newey_west_t([np.nan, 1.0, np.nan, 2.0], 1))
    three = sv.newey_west_t([1.0, 2.0, 4.0], 1)
    assert not any(math.isnan(v) for v in three)
    assert sv.newey_west_t([], 1) == pytest.approx(nan3, nan_ok=True)
    # NaNs are dropped, not propagated
    assert sv.newey_west_t([1.0, np.nan, 2.0, 4.0, np.nan], 1) == pytest.approx(three)
    # Series and ndarray inputs agree
    assert sv.newey_west_t(pd.Series([1.0, 2.0, 4.0]), 1) == pytest.approx(three)


def test_newey_west_lags_capped_at_n_minus_1():
    x = [1.0, 3.0, 2.0]
    assert sv.newey_west_t(x, 50) == pytest.approx(nw_oracle(np.array(x), 50))
    # the Bartlett weight still uses the *requested* lag count
    assert sv.newey_west_t(x, 50) != pytest.approx(sv.newey_west_t(x, 2))


def test_newey_west_constant_series_does_not_divide_by_zero():
    mu, t, p = sv.newey_west_t(np.full(10, 0.1), 2)
    assert mu == pytest.approx(0.1)
    assert t > 1e100 and p == 0.0
    mu0, t0, p0 = sv.newey_west_t(np.zeros(10), 2)
    assert (mu0, t0) == (0.0, 0.0) and p0 == pytest.approx(1.0)


def test_newey_west_negative_mean_gives_negative_t_positive_p():
    mu, t, p = sv.newey_west_t([-1.0, -2.0, -1.5, -2.5, -1.2], 1)
    assert mu < 0 and t < 0 and 0 < p < 1


# ── bh_fdr ───────────────────────────────────────────────────────────────────
def test_bh_fdr_known_values_any_order():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    q = sv.bh_fdr(p)
    assert q == pytest.approx([0.02, 0.04, 0.04, 0.02])
    perm = np.array([2, 0, 3, 1])
    assert sv.bh_fdr(p[perm]) == pytest.approx(q[perm])


def test_bh_fdr_monotone_enforced_and_clipped():
    # raw BH values 0.03*4/1=.12, .01*4/2... unsorted raw: step-up must take the running min from the top
    p = np.array([0.001, 0.2, 0.021, 0.022])
    q = sv.bh_fdr(p)
    assert q == pytest.approx([0.004, 0.2, 0.0293333333333333, 0.0293333333333333], abs=1e-12)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-15)
    assert np.all(q >= p - 1e-15)
    assert sv.bh_fdr([1.5])[0] == 1.0
    assert sv.bh_fdr([0.0])[0] == 0.0


def test_bh_fdr_known_adjusted_values():
    # statsmodels fdr_bh on [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    want = [0.01, 0.04, 0.084, 0.084, 0.084, 0.1, 0.10571428571, 0.216, 0.216, 0.216]
    assert sv.bh_fdr(p) == pytest.approx(want, abs=1e-9)


def test_bh_fdr_nan_preserved_and_m_counts_only_valid():
    q = sv.bh_fdr([0.01, np.nan, 0.04])
    assert math.isnan(q[1])
    assert q[0] == pytest.approx(0.02) and q[2] == pytest.approx(0.04)
    allnan = sv.bh_fdr([np.nan, np.nan])
    assert np.isnan(allnan).all() and allnan.shape == (2,)
    assert sv.bh_fdr([]).shape == (0,)


# ── deflated_sharpe ──────────────────────────────────────────────────────────
def dsr_oracle(r, n_trials, var):
    r = np.asarray(r, float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    n = max(n_trials, 2)
    sr0 = math.sqrt(max(var, 0)) * ((1 - 0.5772156649) * stats.norm.ppf(1 - 1 / n)
                                    + 0.5772156649 * stats.norm.ppf(1 - 1 / (n * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return sr, stats.norm.cdf(z)


def test_deflated_sharpe_matches_oracle():
    rng = np.random.default_rng(6)
    r = rng.standard_t(5, size=80) * 0.02 + 0.004
    for n_trials, var in [(50, 0.01), (2, 0.0004), (1000, 0.05)]:
        assert sv.deflated_sharpe(r, n_trials, var) == pytest.approx(dsr_oracle(r, n_trials, var), rel=1e-12)


def test_deflated_sharpe_more_trials_deflates_more():
    rng = np.random.default_rng(7)
    r = rng.normal(0.01, 0.02, 120)
    lo = sv.deflated_sharpe(r, 5, 0.02)[1]
    hi = sv.deflated_sharpe(r, 5000, 0.02)[1]
    assert hi < lo


def test_deflated_sharpe_n_trials_floor_and_negative_var_clamped():
    rng = np.random.default_rng(8)
    r = rng.normal(0.01, 0.02, 60)
    assert sv.deflated_sharpe(r, 0, 0.02) == pytest.approx(sv.deflated_sharpe(r, 2, 0.02))
    assert sv.deflated_sharpe(r, 1, 0.02) == pytest.approx(sv.deflated_sharpe(r, 2, 0.02))
    assert sv.deflated_sharpe(r, 3, 0.02) != pytest.approx(sv.deflated_sharpe(r, 2, 0.02))
    # negative variance -> treated as zero -> no deflation (sr0 = 0)
    assert sv.deflated_sharpe(r, 99, -1.0) == pytest.approx(sv.deflated_sharpe(r, 99, 0.0))
    sr, dsr = sv.deflated_sharpe(r, 99, 0.0)
    assert dsr == pytest.approx(stats.norm.cdf(dsr_z0(r)))


def dsr_z0(r):
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    return sr * math.sqrt(len(r) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)


def test_deflated_sharpe_degenerate_inputs():
    nan2 = (float("nan"), float("nan"))
    assert sv.deflated_sharpe(np.arange(9, dtype=float), 10, 0.1) == pytest.approx(nan2, nan_ok=True)
    assert sv.deflated_sharpe(np.zeros(30), 10, 0.1) == pytest.approx(nan2, nan_ok=True)
    ten = np.array([0.01, -0.02, 0.03, 0.0, 0.015, -0.01, 0.02, 0.005, -0.005, 0.012])
    sr, dsr = sv.deflated_sharpe(ten, 10, 0.1)               # T == 10 is allowed
    assert math.isfinite(sr) and 0 <= dsr <= 1
    # NaNs dropped before the length check
    withnan = np.concatenate([ten[:9], [np.nan] * 5])
    assert math.isnan(sv.deflated_sharpe(withnan, 10, 0.1)[0])
    assert sv.deflated_sharpe(pd.Series(ten), 10, 0.1) == pytest.approx(sv.deflated_sharpe(ten, 10, 0.1))


# ── neutralize ───────────────────────────────────────────────────────────────
def test_neutralize_removes_exposure_and_keeps_residual_orthogonal():
    rng = np.random.default_rng(9)
    expo = panel(rng.normal(size=(4, 60)))
    noise = panel(rng.normal(size=(4, 60)))
    sig = expo * 2 + noise * 0.5
    res = sv.neutralize(sig, [expo])
    rx = expo.rank(axis=1, pct=True)
    for d in sig.index:
        r = res.loc[d].to_numpy()
        assert abs(np.nanmean(r)) < 1e-12
        assert abs(np.corrcoef(r, rx.loc[d].to_numpy())[0, 1]) < 1e-10
    # a signal that IS the exposure leaves only round-off
    assert np.nanmax(np.abs(sv.neutralize(expo * 5 - 1, [expo]).to_numpy())) < 1e-12


def test_neutralize_matches_lstsq_residual_and_shape():
    rng = np.random.default_rng(10)
    e1, e2 = panel(rng.normal(size=(2, 30))), panel(rng.normal(size=(2, 30)))
    s = panel(rng.normal(size=(2, 30)))
    res = sv.neutralize(s, [e1, e2])
    assert res.shape == s.shape and (res.index == s.index).all()
    y = s.rank(axis=1, pct=True).iloc[0].to_numpy()
    X = np.column_stack([np.ones(30), e1.rank(axis=1, pct=True).iloc[0], e2.rank(axis=1, pct=True).iloc[0]])
    want = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    assert res.iloc[0].to_numpy() == pytest.approx(want, abs=1e-12)


def test_neutralize_min_observation_boundary():
    rng = np.random.default_rng(11)
    s = panel(rng.normal(size=(2, 20)))
    e = panel(rng.normal(size=(2, 20)))
    s.iloc[0, 12:] = np.nan      # 12 usable names == 2 columns + 10 -> regress
    s.iloc[1, 11:] = np.nan      # 11 usable -> skipped
    res = sv.neutralize(s, [e])
    assert res.iloc[0].notna().sum() == 12
    assert res.iloc[1].isna().all()
    # exposure NaN also removes a name from the fit
    e2 = e.copy()
    e2.iloc[0, 0] = np.nan
    assert sv.neutralize(s, [e2]).iloc[0].isna().all()      # 11 usable names left -> skipped


def test_neutralize_aligns_exposure_to_signal_shape():
    rng = np.random.default_rng(12)
    s = panel(rng.normal(size=(3, 30)))
    e = panel(rng.normal(size=(3, 30))).iloc[:, :25]      # fewer columns than the signal
    res = sv.neutralize(s, [e])
    assert res.shape == s.shape
    assert res.iloc[:, 25:].isna().all().all()
    assert res.iloc[:, :25].notna().all().all()


# ── fama_macbeth ─────────────────────────────────────────────────────────────
def fm_inputs(n_dates=40, n=80, seed=13):
    rng = np.random.default_rng(seed)
    a, b = panel(rng.normal(size=(n_dates, n))), panel(rng.normal(size=(n_dates, n)))
    fwd = a * 0.8 + panel(rng.normal(size=(n_dates, n))) * 0.5
    return a, b, fwd


def test_fama_macbeth_recovers_the_priced_signal():
    a, b, fwd = fm_inputs()
    out = sv.fama_macbeth({"a": a, "b": b}, fwd, lags=2)
    assert list(out.columns) == ["signal", "fm_coef", "fm_t", "fm_p", "n_dates"]
    assert list(out.signal) == ["a", "b"]
    ra, rb = out.iloc[0], out.iloc[1]
    assert ra.fm_coef > 0.3 and ra.fm_t > 10 and ra.fm_p < 1e-6
    assert abs(rb.fm_coef) < 0.1 and abs(rb.fm_t) < 3
    assert ra.n_dates == 40 == rb.n_dates


def test_fama_macbeth_matches_manual_per_date_ols():
    a, b, fwd = fm_inputs(n_dates=10, n=60, seed=14)
    out = sv.fama_macbeth({"a": a, "b": b}, fwd, lags=0)
    z = lambda df: ((df.rank(axis=1, pct=True)).sub(df.rank(axis=1, pct=True).mean(axis=1), axis=0)
                    .div(df.rank(axis=1, pct=True).std(axis=1), axis=0))
    za, zb, zy = z(a), z(b), z(fwd)
    coefs = []
    for d in fwd.index:
        X = np.column_stack([np.ones(60), za.loc[d], zb.loc[d]])
        coefs.append(np.linalg.lstsq(X, zy.loc[d].to_numpy(), rcond=None)[0][1:])
    coefs = np.array(coefs)
    assert out.fm_coef.to_numpy() == pytest.approx(coefs.mean(axis=0), abs=1e-12)
    want_t = [sv.newey_west_t(coefs[:, j], 0)[1] for j in range(2)]
    assert out.fm_t.to_numpy() == pytest.approx(want_t)


def test_fama_macbeth_with_missing_values_matches_manual_ols_on_complete_rows():
    a, b, fwd = fm_inputs(n_dates=8, n=70, seed=40)
    a, fwd = a.copy(), fwd.copy()
    a.iloc[:, :9] = np.nan                       # different signals have different numbers of names
    fwd.iloc[:, 60:] = np.nan
    out = sv.fama_macbeth({"a": a, "b": b}, fwd, lags=0)

    def z(df):
        r = df.rank(axis=1, pct=True)
        return r.sub(r.mean(axis=1), axis=0).div(r.std(axis=1), axis=0)          # sample std, skipping NaN
    za, zb, zy = z(a), z(b), z(fwd)
    coefs = []
    for d in fwd.index:
        ok = za.loc[d].notna() & zb.loc[d].notna() & zy.loc[d].notna()
        X = np.column_stack([np.ones(ok.sum()), za.loc[d][ok], zb.loc[d][ok]])
        coefs.append(np.linalg.lstsq(X, zy.loc[d][ok].to_numpy(), rcond=None)[0][1:])
    assert out.fm_coef.to_numpy() == pytest.approx(np.mean(coefs, axis=0), rel=1e-10)


def test_fama_macbeth_min_names_boundary():
    a, b, fwd = fm_inputs(n_dates=6, n=80, seed=15)
    a, b = a.copy(), b.copy()
    a.iloc[0, 50:] = np.nan          # date 0: exactly 50 usable -> regressed
    a.iloc[1, 49:] = np.nan          # date 1: 49 usable -> NaN coefficient row
    out = sv.fama_macbeth({"a": a, "b": b}, fwd, lags=1)
    assert (out.n_dates == 5).all()
    out_lo = sv.fama_macbeth({"a": a, "b": b}, fwd, lags=1, min_names=40)
    assert (out_lo.n_dates == 6).all()


def test_fama_macbeth_min_names_floor_is_signals_plus_ten():
    rng = np.random.default_rng(16)
    sigs = {f"s{i}": panel(rng.normal(size=(5, 30))) for i in range(25)}
    fwd = panel(rng.normal(size=(5, 30)))
    out = sv.fama_macbeth(sigs, fwd, lags=1, min_names=1)     # 30 names < 25 + 10 -> never regress
    assert (out.n_dates == 0).all() and out.fm_coef.isna().all()


def test_fama_macbeth_aligns_signal_columns_to_forward_returns():
    a, b, fwd = fm_inputs(n_dates=8, n=70, seed=17)
    shuffled = a[list(reversed(a.columns))]
    o1 = sv.fama_macbeth({"a": a, "b": b}, fwd, lags=1)
    o2 = sv.fama_macbeth({"a": shuffled, "b": b}, fwd, lags=1)
    assert o1.fm_coef.to_numpy() == pytest.approx(o2.fm_coef.to_numpy())


# ── quantile_spread_turnover ─────────────────────────────────────────────────
def test_quantile_spread_static_signal_has_zero_turnover():
    sig = panel(np.tile(np.arange(50.0), (4, 1)))
    out = sv.quantile_spread_turnover(sig, sig)
    assert out["n_periods"] == 4
    assert out["ls_spread"] == pytest.approx(44.5 - 4.5)        # mean(40..49) - mean(0..9)
    assert out["long_excess"] == pytest.approx(44.5 - 24.5)
    assert out["turnover_long"] == 0.0 and out["turnover_ls"] == 0.0
    assert math.isnan(out["breakeven_rt_cost_ls"]) and math.isnan(out["breakeven_rt_cost_long"])
    assert out["ls_series"].tolist() == [40.0] * 4


def test_quantile_spread_half_rotation_turnover_and_breakeven():
    base = np.arange(50.0)
    sig = panel(np.vstack([base, (np.arange(50) + 5) % 50, base]))
    fwd = panel(np.tile(base, (3, 1)))
    out = sv.quantile_spread_turnover(sig, fwd)
    assert out["turnover_long"] == pytest.approx(0.5)
    assert out["turnover_ls"] == pytest.approx(0.5)
    assert out["breakeven_rt_cost_ls"] == pytest.approx(out["ls_spread"] / (2 * 0.5))
    assert out["breakeven_rt_cost_long"] == pytest.approx(out["long_excess"] / 0.5)
    # first period uses the same signal as fwd: spread = 40; second uses a rotated signal
    assert out["ls_series"][0] == pytest.approx(40.0)
    assert out["n_periods"] == 3 and math.isfinite(out["ls_sharpe"])


def test_quantile_spread_full_reversal_turnover_one_and_decile_cut():
    base = np.arange(50.0)
    sig = panel(np.vstack([base, base[::-1]]))
    fwd = panel(np.tile(base, (2, 1)))
    out = sv.quantile_spread_turnover(sig, fwd)
    assert out["turnover_long"] == 1.0 and out["turnover_ls"] == 1.0
    assert out["ls_series"].tolist() == pytest.approx([40.0, -40.0])
    q10 = sv.quantile_spread_turnover(sig, fwd, q=0.1)
    assert q10["ls_series"][0] == pytest.approx(47.0 - 2.0)     # top 5 mean 47, bottom 5 mean 2


def test_quantile_cut_includes_boundary_name_in_bottom_only():
    # 50 names, q=.2: bottom = ranks 1..10 (pct <= .2), top = ranks 41..50 (pct > .8)
    base = np.arange(50.0)
    fwd = panel(np.tile(np.where(base < 10, -1.0, np.where(base >= 40, 1.0, 0.0)), (1, 1)))
    out = sv.quantile_spread_turnover(panel(base[None, :]), fwd)
    assert out["ls_spread"] == pytest.approx(2.0)               # exactly 10 names per tail


def test_quantile_spread_skips_thin_dates_and_resets_turnover_chain():
    base = np.arange(60.0)
    thin = base.copy()
    thin[49:] = np.nan                                          # 49 valid names
    exact = base.copy()
    exact[50:] = np.nan                                         # 50 valid names
    sig = panel(np.vstack([base, thin, base]))
    out = sv.quantile_spread_turnover(sig, sig)
    assert out["n_periods"] == 2                                # thin date dropped
    assert math.isnan(out["turnover_long"])                     # chain broken by the skipped date
    assert math.isnan(out["turnover_ls"])
    assert out["ls_sharpe"] != out["ls_sharpe"]                 # <3 periods -> NaN
    out50 = sv.quantile_spread_turnover(panel(np.vstack([exact])), panel(np.vstack([exact])))
    assert out50["n_periods"] == 1
    lowmin = sv.quantile_spread_turnover(sig, sig, min_names=49)
    assert lowmin["n_periods"] == 3


def test_quantile_spread_empty_and_sharpe_threshold():
    sig = panel(np.full((2, 10), np.nan))
    out = sv.quantile_spread_turnover(sig, sig)
    assert out["n_periods"] == 0 and math.isnan(out["ls_spread"]) and math.isnan(out["long_excess"])
    assert math.isnan(out["ls_sharpe"]) and out["ls_series"].size == 0
    rng = np.random.default_rng(18)
    three = panel(rng.normal(size=(3, 50)))
    two = panel(rng.normal(size=(2, 50)))
    assert math.isfinite(sv.quantile_spread_turnover(three, three * 0 + rng.normal(size=(3, 50)))["ls_sharpe"])
    assert math.isnan(sv.quantile_spread_turnover(two, two)["ls_sharpe"])


def test_quantile_spread_sharpe_value():
    rng = np.random.default_rng(19)
    sig = panel(rng.normal(size=(8, 60)))
    fwd = panel(sig.to_numpy() + rng.normal(size=(8, 60)))
    out = sv.quantile_spread_turnover(sig, fwd)
    ls = out["ls_series"]
    assert out["ls_sharpe"] == pytest.approx(ls.mean() / ls.std(ddof=1))
    assert out["ls_spread"] == pytest.approx(ls.mean())


# ── purged_walk_forward ──────────────────────────────────────────────────────
def wf_panels(n_dates=150, n=60, seed=20):
    rng = np.random.default_rng(seed)
    fwd = panel(rng.normal(size=(n_dates, n)))
    good = fwd + panel(rng.normal(size=(n_dates, n))) * 1.0
    junk = panel(rng.normal(size=(n_dates, n)))
    return fwd, good, junk


def test_purged_walk_forward_too_short_returns_empty():
    fwd, good, junk = wf_panels(n_dates=120)
    # 120 - 104 = 16 < 5 folds * 4
    assert sv.purged_walk_forward({"g": good}, fwd, embargo=3).empty
    fwd, good, junk = wf_panels(n_dates=124)
    assert not sv.purged_walk_forward({"g": good, "j": junk}, fwd, embargo=3).empty   # 20 == 5 * 4 -> allowed
    fwd, good, junk = wf_panels(n_dates=123)
    assert sv.purged_walk_forward({"g": good, "j": junk}, fwd, embargo=3).empty


def test_purged_walk_forward_folds_windows_and_columns():
    fwd, good, junk = wf_panels()
    out = sv.purged_walk_forward({"good": good, "junk": junk}, fwd, embargo=3)
    assert list(out.columns) == ["fold", "test_start", "test_end", "best_single_name",
                                 "oos_ic_best_single", "oos_ic_ic_weighted", "oos_ic_ridge"]
    assert out.fold.tolist() == [1, 2, 3, 4, 5]
    edges = np.linspace(104, 150, 6).astype(int)
    dates = fwd.index
    assert out.test_start.tolist() == [str(dates[e].date()) for e in edges[:-1]]
    assert out.test_end.tolist() == [str(dates[e - 1].date()) for e in edges[1:]]
    assert (out.best_single_name == "good").all()
    assert (out.oos_ic_best_single > 0.3).all() and (out.oos_ic_ic_weighted > 0.3).all()
    assert (out.oos_ic_ridge > 0.3).all()


def test_purged_walk_forward_best_single_keeps_sign_of_negative_signal():
    fwd, good, junk = wf_panels(seed=21)
    out = sv.purged_walk_forward({"inv": -good, "junk": junk}, fwd, embargo=2)
    assert (out.best_single_name == "inv").all()
    assert (out.oos_ic_best_single > 0.3).all()                 # sign flipped back to positive
    assert (out.oos_ic_ic_weighted > 0.3).all()                 # weights carry the negative IC


def test_purged_walk_forward_n_folds_and_min_train_arguments():
    fwd, good, junk = wf_panels()
    out3 = sv.purged_walk_forward({"g": good, "j": junk}, fwd, embargo=2, n_folds=3)
    assert out3.fold.tolist() == [1, 2, 3]
    out_early = sv.purged_walk_forward({"g": good, "j": junk}, fwd, embargo=2, min_train=60)
    assert out_early.test_start.iloc[0] == str(fwd.index[60].date())


def test_purged_walk_forward_skips_fold_when_training_below_52_dates():
    fwd, good, junk = wf_panels(n_dates=100)
    # min_train=60, embargo=10: fold 1 trains on 50 dates (<52) -> skipped; later folds kept
    out = sv.purged_walk_forward({"g": good, "j": junk}, fwd, embargo=10, n_folds=5, min_train=60)
    assert out.fold.min() == 2 and len(out) == 4
    # embargo 8 -> exactly 52 training dates -> fold 1 kept
    out2 = sv.purged_walk_forward({"g": good, "j": junk}, fwd, embargo=8, n_folds=5, min_train=60)
    assert out2.fold.min() == 1 and len(out2) == 5


def test_purged_walk_forward_embargo_drops_the_last_training_dates():
    # `late` predicts perfectly only on the dates inside the embargo window before fold 1's test
    # start; `steady` has a modest real IC everywhere. With purging the late signal never gets
    # to look good in training.
    rng = np.random.default_rng(22)
    n_dates, n = 150, 60
    fwd = panel(rng.normal(size=(n_dates, n)))
    steady = fwd * 0.25 + panel(rng.normal(size=(n_dates, n)))
    late = panel(rng.normal(size=(n_dates, n)))
    late.iloc[64:104] = fwd.iloc[64:104]
    out = sv.purged_walk_forward({"late": late, "steady": steady}, fwd, embargo=40, n_folds=1)
    assert out.best_single_name.tolist() == ["steady"]
    out0 = sv.purged_walk_forward({"late": late, "steady": steady}, fwd, embargo=0, n_folds=1)
    assert out0.best_single_name.tolist() == ["late"]


def test_purged_walk_forward_nan_best_ic_never_chosen():
    rng = np.random.default_rng(23)
    fwd = panel(rng.normal(size=(150, 60)))
    dead = panel(np.full((150, 60), np.nan))
    live = fwd + panel(rng.normal(size=(150, 60)))
    out = sv.purged_walk_forward({"dead": dead, "live": live}, fwd, embargo=2, n_folds=2)
    assert (out.best_single_name == "live").all()
    assert (out.oos_ic_best_single > 0.3).all()


# ── ic_health ────────────────────────────────────────────────────────────────
def ic_series(blocks):
    vals = np.concatenate([np.full(n, v) for n, v in blocks]).astype(float)
    vals += 0.01 * np.where(np.arange(len(vals)) % 2 == 0, 1.0, -1.0)
    return pd.Series(vals, index=pd.date_range("2020-01-03", periods=len(vals), freq="W-FRI"))


def test_ic_health_healthy():
    h = sv.ic_health(ic_series([(100, 0.06)]))
    assert h["status"] == "HEALTHY"
    assert set(h) == {"full_ic", "full_t", "recent_ic", "recent_t", "status"}
    assert h["full_ic"] == pytest.approx(0.06, abs=1e-3) and abs(h["full_t"]) > 2


def test_ic_health_decaying_flipped_noise():
    assert sv.ic_health(ic_series([(60, 0.08), (52, 0.01)]))["status"] == "DECAYING"
    flipped = sv.ic_health(ic_series([(60, 0.08), (52, -0.05)]))
    assert flipped["status"] == "FLIPPED" and flipped["recent_ic"] < 0 < flipped["full_ic"]
    assert sv.ic_health(ic_series([(60, 0.0), (60, 0.0)]))["status"] == "NOISE"


def test_ic_health_opposite_sign_but_insignificant_recent_is_decaying():
    rng = np.random.default_rng(24)
    ic = pd.Series(np.concatenate([np.full(90, 0.08), rng.normal(-0.002, 0.3, 52)]))
    h = sv.ic_health(ic)
    assert abs(h["recent_t"]) < 2 and h["recent_ic"] < 0
    assert h["status"] == "DECAYING"


def test_ic_health_half_magnitude_boundary_is_healthy():
    ic = pd.Series([3.0] * 10 + [1.0] * 10)         # full 2.0, recent 1.0 == 0.5 * full
    assert sv.ic_health(ic, window=10)["status"] == "HEALTHY"
    ic2 = pd.Series([3.0] * 10 + [0.99] * 10)
    assert sv.ic_health(ic2, window=10)["status"] == "DECAYING"


def test_ic_health_window_uses_last_non_nan_values():
    ic = ic_series([(60, 0.08), (30, 0.0)])
    padded = pd.concat([ic, pd.Series([np.nan] * 5, index=pd.date_range("2030-01-04", periods=5, freq="W-SAT"))])
    assert sv.ic_health(padded, window=30) == sv.ic_health(ic, window=30)
    h10 = sv.ic_health(ic, window=10)
    h52 = sv.ic_health(ic, window=52)
    assert h10["recent_ic"] == pytest.approx(0.0, abs=2e-3)
    assert h52["recent_ic"] > h10["recent_ic"]


def test_ic_health_lags_argument_changes_the_t_stat():
    ic = ic_series([(100, 0.04)])
    assert sv.ic_health(ic, lags=1)["full_t"] != sv.ic_health(ic, lags=5)["full_t"]
    assert sv.ic_health(ic, lags=5)["full_t"] == pytest.approx(sv.newey_west_t(ic, 5)[1])


def test_ic_health_all_nan_is_noise():
    h = sv.ic_health(pd.Series([np.nan] * 10))
    assert h["status"] == "NOISE" and math.isnan(h["full_t"]) and math.isnan(h["recent_ic"])
