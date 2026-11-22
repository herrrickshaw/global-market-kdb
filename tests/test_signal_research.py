"""Offline tests for python/upstream/signal_research.py.

Everything runs on small synthetic panels; the Parquet warehouse is a few files written
into tmp_path with DuckDB, and ``~`` is redirected into tmp_path.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "upstream"))
import signal_validation as sv  # noqa: E402

# A private copy of the module: study_py.run() (used by test_parity.py when q is installed, and by
# test_study_py_offline.py) rewires signal_research's globals, and these tests must not see that.
_spec = importlib.util.spec_from_file_location("signal_research_under_test",
                                               ROOT / "python" / "upstream" / "signal_research.py")
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)

N_DAYS, N_NAMES = 1000, 70
DAYS = pd.bdate_range("2018-01-01", periods=N_DAYS)          # starts on a Monday: Fridays are idx % 5 == 4
START = "2019-06-07"                                         # a Friday
SIGNALS = ["rev5", "rsi_os", "mom21", "mom126", "mom12_1", "trend", "golden", "hi52",
           "lowvol", "illiq", "volsurge"]


def make_panels(seed=0, n_days=N_DAYS, n=N_NAMES):
    rng = np.random.default_rng(seed)
    days = DAYS[:n_days]
    cols = [f"S{i:02d}" for i in range(n)]
    r = rng.normal(0.0003, 0.015, (n_days, n))
    close = pd.DataFrame(50 * np.exp(np.cumsum(r, axis=0)), index=days, columns=cols)
    high = close * (1 + np.abs(rng.normal(0, 0.005, (n_days, n))))
    vol = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n)), index=days, columns=cols)
    return {"Close": close.astype("float32"), "High": high.astype("float32"),
            "Volume": vol.astype("float32")}


@pytest.fixture
def small_universe(monkeypatch):
    monkeypatch.setattr(sr, "START", START)
    monkeypatch.setitem(sr.UNIVERSE_N, "US", N_NAMES)


@pytest.fixture(scope="module")
def base_panels():
    return make_panels()


# ── tiny helpers ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("h,want", [(1, 1), (5, 1), (6, 2), (10, 2), (21, 5), (63, 13), (0, 1)])
def test_lags_for(h, want):
    assert sr.lags_for(h) == want


def test_fmt_blank_for_missing_and_non_finite():
    assert sr.fmt(None) == ""
    assert sr.fmt(float("nan")) == ""
    assert sr.fmt(float("inf")) == "" and sr.fmt(float("-inf")) == ""
    assert sr.fmt(np.float64("nan")) == ""
    assert sr.fmt(0.0) == "0.000"
    assert sr.fmt(1.23456) == "1.235"
    assert sr.fmt(1.23456, 1) == "1.2"
    assert sr.fmt(7, 2) == "7.00"                                    # ints are formatted, not blanked
    assert sr.fmt(2.0, 0) == "2"


def test_ohlcv_root_cli_value_wins_and_needs_no_warehouse(monkeypatch):
    monkeypatch.setitem(sys.modules, "market_paths", None)           # importing it would fail
    assert sr.ohlcv_root("/some/where") == Path("/some/where")


def test_ohlcv_root_falls_back_to_pipeline_seam(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "market_paths.py").write_text(
        "def resolve(name):\n    return '/warehouse/' + name\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "market_paths", raising=False)
    try:
        assert sr.ohlcv_root(None) == Path("/warehouse/ohlcv")
        assert sys.path[0] == str(scripts)
    finally:
        sys.modules.pop("market_paths", None)


def test_ohlcv_root_empty_string_means_not_given(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "market_paths.py").write_text("def resolve(name):\n    return '/seam/' + name\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "market_paths", raising=False)
    try:
        assert sr.ohlcv_root("") == Path("/seam/ohlcv")
    finally:
        sys.modules.pop("market_paths", None)


def test_mailer_rules_reads_both_files_and_tolerates_bad_ones(tmp_path, monkeypatch):
    base = tmp_path / "market-pipeline" / "code" / "python_files" / "cache_seed"
    base.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert sr.mailer_rules() == {"zone_rules.json": {}, "zone_regime.json": {}}       # both missing
    (base / "zone_rules.json").write_text('{"US": {"x": 1}}')
    (base / "zone_regime.json").write_text("{not json")
    got = sr.mailer_rules()
    assert got == {"zone_rules.json": {"US": {"x": 1}}, "zone_regime.json": {}}
    (base / "zone_regime.json").write_text('{"IN": {"bull_rule": "revert"}}')
    assert sr.mailer_rules()["zone_regime.json"] == {"IN": {"bull_rule": "revert"}}


# ── load_market (DuckDB over a Parquet tree) ─────────────────────────────────
def write_parquet(root, mkt, year, rows):
    d = root / mkt
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["Date", "Symbol", "Close", "High", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    con = duckdb.connect()
    con.register("df", df)
    con.execute(f"COPY (SELECT * FROM df) TO '{d / f'year={year}.parquet'}' (FORMAT PARQUET)")


def rows_for(year, sym, close, vol, days=4):
    return [(f"{year}-03-0{d + 1}", sym, close, close * 1.1, vol) for d in range(days)]


def test_load_market_keeps_names_in_top_n_of_any_year(tmp_path, monkeypatch):
    monkeypatch.setitem(sr.UNIVERSE_N, "US", 2)               # int(2 * 1.3) == 2 names per year
    # 2020 dollar volume: A 1000 > B 500 > C 100 > D 50; 2021: C 900 > A 800 > B 20 > D 10
    r20 = sum((rows_for(2020, s, 10.0, v) for s, v in [("A", 100), ("B", 50), ("C", 10), ("D", 5)]), [])
    r21 = sum((rows_for(2021, s, 10.0, v) for s, v in [("C", 90), ("A", 80), ("B", 2), ("D", 1)]), [])
    write_parquet(tmp_path, "US", 2020, r20)
    write_parquet(tmp_path, "US", 2021, r21)
    p = sr.load_market(tmp_path, "US")
    assert set(p) == {"Close", "High", "Volume"}
    assert list(p["Close"].columns) == ["A", "B", "C"]         # D was never in the top 2
    assert p["Close"].shape == (8, 3)                          # 4 days in each of the two years
    for k in p:
        assert p[k].dtypes.eq("float32").all()
        assert p[k].index.is_monotonic_increasing
        assert isinstance(p[k].index, pd.DatetimeIndex)
    assert p["High"]["A"].iloc[0] == pytest.approx(11.0)


def test_load_market_pool_is_int_of_1_3_times_n(tmp_path, monkeypatch):
    monkeypatch.setitem(sr.UNIVERSE_N, "US", 5)               # int(6.5) == 6 names kept per year
    rows = sum((rows_for(2020, f"S{i}", 10.0, 100 - i) for i in range(8)), [])
    write_parquet(tmp_path, "US", 2020, rows)
    assert list(sr.load_market(tmp_path, "US")["Close"].columns) == [f"S{i}" for i in range(6)]
    monkeypatch.setitem(sr.UNIVERSE_N, "US", 2)               # int(2.6) == 2, not round() == 3
    assert list(sr.load_market(tmp_path, "US")["Close"].columns) == ["S0", "S1"]


def test_load_market_filters_nonpositive_close_and_ignores_zero_volume_in_ranking(tmp_path, monkeypatch):
    monkeypatch.setitem(sr.UNIVERSE_N, "US", 1)
    rows = rows_for(2020, "A", 10.0, 100)
    rows += [("2020-03-10", "A", 0.0, 0.0, 100), ("2020-03-11", "A", -5.0, -5.0, 100)]
    rows += [("2020-03-12", "A", 10.0, 11.0, 0)]               # zero volume: ranked out, but the bar is kept
    rows += rows_for(2020, "Z", 10.0, 0)                       # never trades: no dollar volume -> not ranked
    write_parquet(tmp_path, "US", 2020, rows)
    p = sr.load_market(tmp_path, "US")
    assert list(p["Close"].columns) == ["A"]
    assert (p["Close"]["A"].dropna() > 0).all()
    assert len(p["Close"]) == 5
    assert p["Volume"]["A"].loc["2020-03-12"] == 0.0


def test_load_market_strips_etf_like_symbols_for_india_only(tmp_path, monkeypatch):
    monkeypatch.setitem(sr.UNIVERSE_N, "IN", 50)
    monkeypatch.setitem(sr.UNIVERSE_N, "US", 50)
    syms = ["TCS", "GOLDBEES", "niftyetf", "LIQUIDCASE", "GILT5Y", "SENSEXADD", "NIFTYBEES", "INFY"]
    rows = sum((rows_for(2020, s, 10.0, 100) for s in syms), [])
    write_parquet(tmp_path, "IN", 2020, rows)
    write_parquet(tmp_path, "US", 2020, rows)
    assert list(sr.load_market(tmp_path, "IN")["Close"].columns) == ["INFY", "TCS"]
    assert len(sr.load_market(tmp_path, "US")["Close"].columns) == 8


# ── build_signals ────────────────────────────────────────────────────────────
def special_panels():
    """70 random names plus engineered columns (each overwrites one random name)."""
    p = make_panels(seed=3)
    C, H, V = p["Close"].copy(), p["High"].copy(), p["Volume"].copy()
    cols = list(C.columns)
    jump_a, jump_b = 700, 704                      # idx 700 is a Monday, 704 a Friday
    names = {"LATE": cols[0], "JUMPA": cols[1], "JUMPB": cols[2], "HALF": cols[3], "DBL": cols[4],
             "GAP3": cols[5], "GAP4": cols[6], "ZEROV": cols[7]}
    # LATE lists at row 400 (Fridays are idx 4 mod 5: row 639 is its first with 240 bars of history)
    C.iloc[:400, 0] = np.nan; H.iloc[:400, 0] = np.nan; V.iloc[:400, 0] = np.nan
    for name, at in (("JUMPA", jump_a), ("JUMPB", jump_b)):
        j = cols.index(names[name])
        C.iloc[at:, j] *= np.float32(0.4); H.iloc[at:, j] *= np.float32(0.4)       # -60% in one day
    for name, before, after in (("HALF", 32.0, 16.0), ("DBL", 16.0, 32.0)):
        j = cols.index(names[name])
        C.iloc[:jump_b, j] = before; C.iloc[jump_b:, j] = after
        H.iloc[:, j] = C.iloc[:, j] * np.float32(1.0)
    C.iloc[697:700, 5] = np.nan                       # 3-day hole ending on Friday 699 -> forward-filled
    C.iloc[696:700, 6] = np.nan                       # 4-day hole -> Friday 699 stays empty
    V.iloc[::10, 7] = 0
    return {"Close": C, "High": H, "Volume": V}, names


@pytest.fixture(scope="module")
def special():
    return special_panels()


@pytest.fixture
def built_special(special, monkeypatch):
    monkeypatch.setattr(sr, "START", START)
    monkeypatch.setitem(sr.UNIVERSE_N, "US", N_NAMES)
    p, names = special
    return sr.build_signals(p, "US"), names, p


def fridays_from(start):
    d = DAYS[DAYS.dayofweek == 4]
    return d[d >= pd.Timestamp(start)]


def test_build_signals_formation_dates_are_fridays_from_start_inclusive(built_special):
    (S, F, X, regime), _, _ = built_special
    want = fridays_from(START)
    assert list(S["rev5"].index) == list(want)
    assert S["rev5"].index[0] == pd.Timestamp(START)             # START itself is a formation date
    assert set(S) == set(SIGNALS) | {"zone"}
    assert set(F) == {5, 10, 21, 63} and set(X) == {"mdv", "vol"}
    assert list(regime.index) == list(want)
    for df in list(S.values()) + list(F.values()) + list(X.values()):
        assert df.index.equals(S["rev5"].index)


def test_build_signals_start_is_exclusive_of_earlier_fridays(small_universe, monkeypatch, base_panels):
    monkeypatch.setattr(sr, "START", "2019-06-08")               # the Saturday after
    S, *_ = sr.build_signals(base_panels, "US")
    assert S["rev5"].index[0] == pd.Timestamp("2019-06-14")


def test_build_signals_values_match_independent_pandas(small_universe, base_panels):
    S, F, X, regime = sr.build_signals(base_panels, "US")
    C, H, V = (base_panels[k].astype("float64") for k in ("Close", "High", "Volume"))
    idx = S["rev5"].index
    ex = lambda df: df.loc[idx]
    close32 = base_panels["Close"]
    np.testing.assert_allclose(S["rev5"].to_numpy(), ex(-(close32 / close32.shift(5) - 1)).to_numpy(), rtol=1e-4)
    np.testing.assert_allclose(S["mom21"].to_numpy(), ex(close32 / close32.shift(21) - 1).to_numpy(), rtol=1e-4)
    np.testing.assert_allclose(S["mom126"].to_numpy(), ex(close32 / close32.shift(126) - 1).to_numpy(), rtol=1e-4)
    np.testing.assert_allclose(S["mom12_1"].to_numpy(),
                               ex(close32.shift(21) / close32.shift(252) - 1).to_numpy(), rtol=1e-4)
    e50, e200 = close32.ewm(span=50, adjust=False).mean(), close32.ewm(span=200, adjust=False).mean()
    np.testing.assert_allclose(S["trend"].to_numpy(), ex(close32 / e50 - 1).to_numpy(), rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(S["golden"].to_numpy(), ex(e50 / e200 - 1).to_numpy(), rtol=1e-3, atol=1e-6)
    hi = base_panels["High"].rolling(252, min_periods=200).max()
    np.testing.assert_allclose(S["hi52"].to_numpy(), ex(close32 / hi).to_numpy(), rtol=1e-4)
    r1 = close32.pct_change(fill_method=None)
    np.testing.assert_allclose(S["lowvol"].to_numpy(),
                               ex(-r1.rolling(63, min_periods=50).std()).to_numpy(), rtol=1e-3, atol=1e-7)
    vr = base_panels["Volume"].rolling(20, min_periods=15).mean() / base_panels["Volume"].rolling(126, min_periods=100).mean()
    np.testing.assert_allclose(S["volsurge"].to_numpy(), ex(np.log(vr)).to_numpy(), rtol=1e-3, atol=1e-6)
    dv = close32 * base_panels["Volume"]
    illiq = (r1.abs() / dv).rolling(63, min_periods=50).mean()
    np.testing.assert_allclose(S["illiq"].to_numpy(), ex(illiq).to_numpy(), rtol=1e-3)
    # RSI: oversold = high, so the signal is the negated 0..100 index
    assert (S["rsi_os"].stack() <= 0).all() and (S["rsi_os"].stack() >= -100).all()
    up = r1.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-r1).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    np.testing.assert_allclose(S["rsi_os"].to_numpy(), ex(-(100 - 100 / (1 + up / dn))).to_numpy(), rtol=1e-3)
    # market-neutral exposures
    mdv = dv.rolling(63, min_periods=40).median()
    np.testing.assert_allclose(X["mdv"].to_numpy(), ex(mdv).to_numpy(), rtol=1e-3)
    np.testing.assert_allclose(X["vol"].to_numpy(), ex(r1.rolling(63, min_periods=50).std()).to_numpy(), rtol=1e-3)


def test_build_signals_forward_returns_enter_next_close(small_universe, base_panels):
    S, F, X, regime = sr.build_signals(base_panels, "US")
    C = base_panels["Close"]
    for h in (5, 10, 21, 63):
        want = (C.shift(-(h + 1)) / C.shift(-1) - 1).loc[F[h].index]
        np.testing.assert_allclose(F[h].to_numpy(), want.to_numpy(), rtol=1e-4, atol=1e-7)
    # the last formation dates have no forward window
    assert F[5].iloc[-1].isna().all() and F[63].iloc[-1].isna().all()
    assert F[5].iloc[-2].isna().all()                  # needs the close 6 rows ahead; only 5 remain
    assert F[5].iloc[-3].notna().all()


def test_build_signals_zone_state(small_universe, base_panels):
    S, *_ = sr.build_signals(base_panels, "US")
    C = base_panels["Close"].astype("float64")
    e20, e50 = C.ewm(span=20, adjust=False).mean(), C.ewm(span=50, adjust=False).mean()
    want = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    want[(C > e20) & (e20 > e50)] = 1.0
    want[C < e50] = -1.0
    got = S["zone"]
    sample = want.loc[got.index]
    agree = (got.to_numpy() == sample.to_numpy()).mean()
    assert agree > 0.995                                   # float32 vs float64 EMA ties only
    assert set(np.unique(got.stack())) == {-1.0, 0.0, 1.0}


def test_build_signals_history_gate_needs_240_bars(built_special):
    (S, *_), names, _ = built_special
    late = names["LATE"]
    first = DAYS[639]                                       # exactly 240 bars of history
    assert S["mom21"][late].loc[DAYS[634]] != S["mom21"][late].loc[DAYS[634]]       # 235 bars: out
    assert S["mom21"][late].loc[DAYS[639]] == S["mom21"][late].loc[DAYS[639]]       # 240 bars: in
    assert S["mom21"][late].loc[:DAYS[634]].isna().all()
    assert S["mom21"][late].loc[first:].notna().all()


def test_build_signals_liquidity_gate_keeps_top_n_by_median_dollar_volume(small_universe, monkeypatch, base_panels):
    monkeypatch.setitem(sr.UNIVERSE_N, "US", 50)
    S, F, X, regime = sr.build_signals(base_panels, "US")
    n_ok = S["mom21"].notna().sum(axis=1)
    assert (n_ok == 50).all()
    assert (X["mdv"].notna().sum(axis=1) == 50).all()
    d = S["mom21"].index[10]
    mdv = (base_panels["Close"] * base_panels["Volume"]).rolling(63, min_periods=40).median().loc[d]
    assert set(S["mom21"].loc[d].dropna().index) == set(mdv.sort_values(ascending=False).index[:50])
    assert F[5].notna().sum(axis=1).iloc[:-2].max() == 50


def test_build_signals_gate_rank_boundary_is_inclusive(small_universe, monkeypatch, base_panels):
    monkeypatch.setitem(sr.UNIVERSE_N, "US", 1)
    S, *_ = sr.build_signals(base_panels, "US")
    assert (S["mom21"].notna().sum(axis=1) == 1).all()


def test_build_signals_corporate_action_guard_masks_lookbacks(built_special):
    (S, F, X, regime), names, _ = built_special
    jb, ja = names["JUMPB"], names["JUMPA"]
    # jump on Friday 704: rev5 (6-day look-back) is masked on 704 and 709, free again on 714
    assert S["rev5"][jb].loc[DAYS[699]] == S["rev5"][jb].loc[DAYS[699]]              # before the jump
    assert math.isnan(S["rev5"][jb].loc[DAYS[704]]) and math.isnan(S["rev5"][jb].loc[DAYS[709]])
    assert not math.isnan(S["rev5"][jb].loc[DAYS[714]])
    # mom21 (22-day look-back): masked 704..724, free on 729
    for t in (704, 709, 714, 719, 724):
        assert math.isnan(S["mom21"][jb].loc[DAYS[t]]), t
    assert not math.isnan(S["mom21"][jb].loc[DAYS[729]])
    # trend uses a 60-day look-back, mom126 127
    assert math.isnan(S["trend"][jb].loc[DAYS[759]]) and not math.isnan(S["trend"][jb].loc[DAYS[764]])
    assert math.isnan(S["mom126"][jb].loc[DAYS[829]]) and not math.isnan(S["mom126"][jb].loc[DAYS[834]])
    # a name without a jump is untouched at the same dates
    assert not math.isnan(S["rev5"]["S10"].loc[DAYS[704]])
    # jump on idx 700 (Monday): rev5 masked 700..705 -> Friday 704 masked
    assert math.isnan(S["rev5"][ja].loc[DAYS[704]])


def test_build_signals_forward_window_guard_edges(built_special):
    (S, F, X, regime), names, _ = built_special
    ja = names["JUMPA"]                                      # jump on row 700
    # forward return from t covers closes t+1 .. t+h+1; the jump at 700 is inside it when
    # t+1 <= 700 <= t+h+1  (a window starting at 700 itself still contains the 700 -> 701 step? no:
    # entry is the t+1 close, so the 700 jump is inside when t+1 <= 700).
    assert math.isnan(F[5][ja].loc[DAYS[694]])               # t = J-6 -> J == t+h+1: masked
    assert not math.isnan(F[5][ja].loc[DAYS[689]])           # t = J-11: window ends at 695
    assert math.isnan(F[10][ja].loc[DAYS[689]])              # t = J-11 -> J == t+h+1
    assert not math.isnan(F[10][ja].loc[DAYS[684]])
    # a window that starts after the jump is clean
    assert not math.isnan(F[5][ja].loc[DAYS[704]]) and not math.isnan(F[63][ja].loc[DAYS[704]])
    # but the 63d window formed 60 days before the jump is contaminated
    assert math.isnan(F[63][ja].loc[DAYS[649]]) and not math.isnan(F[63][ja].loc[DAYS[634]])


def test_build_signals_jump_thresholds_are_strict(built_special):
    (S, *_), names, _ = built_special
    half, dbl = names["HALF"], names["DBL"]
    # exactly -50% and exactly +100% are NOT corporate actions; the guard only trips beyond them
    assert not math.isnan(S["rev5"][half].loc[DAYS[704]])
    assert not math.isnan(S["rev5"][dbl].loc[DAYS[704]])
    assert S["rev5"][half].loc[DAYS[704]] == pytest.approx(-(16 / 32 - 1))
    assert S["rev5"][dbl].loc[DAYS[704]] == pytest.approx(-(32 / 16 - 1))


def test_build_signals_forward_fill_is_limited_to_three_days(built_special):
    (S, *_), names, _ = built_special
    gap3, gap4 = names["GAP3"], names["GAP4"]
    assert not math.isnan(S["rev5"][gap3].loc[DAYS[699]])       # 3 missing days: carried forward
    assert math.isnan(S["rev5"][gap4].loc[DAYS[699]])           # 4 missing days: the 4th stays empty


def test_build_signals_zero_volume_days_are_excluded_from_amihud(built_special):
    (S, *_), names, _ = built_special
    zv = S["illiq"][names["ZEROV"]].dropna()
    assert len(zv) > 0 and np.isfinite(zv).all() and (zv > 0).all()


def test_build_signals_regime_flips_at_45_percent_breadth(monkeypatch):
    monkeypatch.setattr(sr, "START", START)
    monkeypatch.setitem(sr.UNIVERSE_N, "US", 60)
    days, t = DAYS, np.arange(N_DAYS)
    def regime_for(n_up):
        cols = [f"S{i:02d}" for i in range(60)]
        close = pd.DataFrame({c: 100 * np.exp((0.002 if i < n_up else -0.002) * t) for i, c in enumerate(cols)},
                             index=days)
        vol = pd.DataFrame({c: np.full(N_DAYS, 1e6 * (1 + i / 100)) for i, c in enumerate(cols)}, index=days)
        return sr.build_signals({"Close": close, "High": close * 1.01, "Volume": vol}, "US")[3]
    assert (regime_for(27) == "bull").all()                    # 27/60 == 0.45 is not < 0.45
    assert (regime_for(26) == "bear").all()
    assert (regime_for(60) == "bull").all() and (regime_for(0) == "bear").all()


def test_build_signals_breadth_counts_only_names_that_pass_the_gate(monkeypatch):
    monkeypatch.setattr(sr, "START", START)
    monkeypatch.setitem(sr.UNIVERSE_N, "US", 30)
    t = np.arange(N_DAYS)
    cols = [f"S{i:02d}" for i in range(60)]
    # the 30 most liquid names: 14 up / 16 down  (14/30 = 0.467 -> bull); the 30 illiquid ones all fall
    drift = [0.002 if i < 14 else -0.002 for i in range(30)] + [-0.002] * 30
    close = pd.DataFrame({c: 100 * np.exp(d * t) for c, d in zip(cols, drift)}, index=DAYS)
    vol = pd.DataFrame({c: np.full(N_DAYS, 1e9 / (1 + i)) for i, c in enumerate(cols)}, index=DAYS)
    S, F, X, regime = sr.build_signals({"Close": close, "High": close * 1.01, "Volume": vol}, "US")
    assert (regime == "bull").all()
    assert (S["mom21"].notna().sum(axis=1) == 30).all()


# ── study_market ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def study(base_panels):
    mp = pytest.MonkeyPatch()
    mp.setattr(sr, "START", START)
    mp.setitem(sr.UNIVERSE_N, "US", N_NAMES)
    mp.setattr(sr, "load_market", lambda root, mkt: {k: v.copy() for k, v in base_panels.items()})
    try:
        res = sr.study_market("US", Path("/nowhere"))
        built = sr.build_signals(base_panels, "US")
    finally:
        mp.undo()
    return res, built


def test_study_market_shapes_and_metadata(study):
    res, (S, F, X, regime) = study
    assert len(res["ic"]) == 12 * 4                                    # 11 signals + zone, 4 horizons
    assert len(res["cost"]) == 11 * 3                                  # no zone; horizons 5, 21, 63
    assert set(res["ic_series"]) == set(SIGNALS) | {"zone"}
    assert set(res["health"]) == set(SIGNALS)
    assert res["n_dates"] == len(regime) == 126
    assert res["universe"] == N_NAMES
    assert res["regime_share_bear"] == pytest.approx(float((regime == "bear").mean()))
    assert res["current_regime"] == regime.iloc[-1]
    assert list(res["fm"].columns)[:2] == ["market", "signal"] and (res["fm"].market == "US").all()
    assert list(res["fm"].signal) == SIGNALS


def test_study_market_ic_rows_recompute_from_the_signals(study):
    res, (S, F, X, regime) = study
    rows = {(r["signal"], r["h"]): r for r in res["ic"]}
    for k, h in [("mom21", 5), ("mom21", 63), ("rev5", 10), ("zone", 21), ("hi52", 10)]:
        L = sr.lags_for(h)
        ic = sv.rank_ic_series(S[k], F[h])
        mu, t, p = sv.newey_west_t(ic, L)
        r = rows[(k, h)]
        assert (r["market"], r["mean_ic"], r["nw_t"], r["p"]) == pytest.approx(("US", mu, t, p))
        assert r["n_dates"] == int(ic.notna().sum())
        assert r["hit"] == pytest.approx(float((ic > 0).mean()))
        assert r["icir_ann"] == pytest.approx(mu / ic.std() * math.sqrt(52 / L))
        for reg in ("bull", "bear"):
            want = sv.newey_west_t(ic[regime == reg], L)
            assert (r[f"{reg}_ic"], r[f"{reg}_t"]) == pytest.approx(want[:2], nan_ok=True)


def test_study_market_neutral_ic_present_for_signals_but_not_for_zone(study):
    res, (S, F, X, regime) = study
    rows = {(r["signal"], r["h"]): r for r in res["ic"]}
    assert "neutral_ic" not in rows[("zone", 10)] and "neutral_t" not in rows[("zone", 10)]
    neut = sv.neutralize(S["mom21"], [X["mdv"], X["vol"]])
    nic = sv.rank_ic_series(neut, F[21])
    want = sv.newey_west_t(nic, sr.lags_for(21))
    assert (rows[("mom21", 21)]["neutral_ic"], rows[("mom21", 21)]["neutral_t"]) == pytest.approx(want[:2])


def test_study_market_ic_series_stored_at_10d_and_health_uses_52w_window(study):
    res, (S, F, X, regime) = study
    for k in ("rev5", "zone"):
        pd.testing.assert_series_equal(res["ic_series"][k], sv.rank_ic_series(S[k], F[10]),
                                       check_names=False)
    want = sv.ic_health(res["ic_series"]["mom126"], window=52, lags=sr.lags_for(10))
    assert res["health"]["mom126"] == want
    assert res["health"]["mom126"]["status"] in {"HEALTHY", "DECAYING", "FLIPPED", "NOISE"}


def test_study_market_fama_macbeth_is_run_at_10d(study):
    res, (S, F, X, regime) = study
    want = sv.fama_macbeth({k: S[k] for k in SIGNALS}, F[10], lags=sr.lags_for(10))
    pd.testing.assert_frame_equal(res["fm"].drop(columns="market").reset_index(drop=True), want)


def test_study_market_cost_rows(study):
    res, (S, F, X, regime) = study
    cost_rt = 0.10 / 100
    seen = set()
    for r in res["cost"]:
        h, k = r["h"], r["signal"]
        seen.add((k, h))
        assert r["market"] == "US" and r["cost_bps"] == pytest.approx(10.0)
        sign = 1.0 if sv.rank_ic_series(S[k], F[h]).mean() >= 0 else -1.0
        assert r["direction"] == ("+" if sign > 0 else "-")
        step = sr.lags_for(h)
        q = sv.quantile_spread_turnover(S[k].iloc[::step] * sign, F[h].iloc[::step])
        ppy = 252 / h
        assert r["ls_spread_pct"] == pytest.approx(100 * q["ls_spread"])
        assert r["turnover_ls"] == pytest.approx(q["turnover_ls"], nan_ok=True)
        assert r["n_periods"] == q["n_periods"]
        assert r["ls_net_ann_pct"] == pytest.approx(
            100 * (q["ls_spread"] - 2 * q["turnover_ls"] * cost_rt) * ppy, nan_ok=True)
        assert r["long_net_ann_pct"] == pytest.approx(
            100 * (q["long_excess"] - q["turnover_long"] * cost_rt) * ppy, nan_ok=True)
        assert r["long_excess_pct"] == pytest.approx(100 * q["long_excess"])
        assert r["ls_sharpe_ann"] == pytest.approx(q["ls_sharpe"] * math.sqrt(ppy), nan_ok=True)
        assert r["breakeven_bps_ls"] == pytest.approx(1e4 * q["breakeven_rt_cost_ls"], nan_ok=True)
        np.testing.assert_allclose(r["_ls_series"], q["ls_series"] - 2 * q["turnover_ls"] * cost_rt)
    assert seen == {(k, h) for k in SIGNALS for h in (5, 21, 63)}


def test_study_market_walk_forward_block(study):
    res, (S, F, X, regime) = study
    want = sv.purged_walk_forward({k: S[k] for k in SIGNALS}, F[10], embargo=sr.lags_for(10) + 1, n_folds=5)
    assert list(res["wf"].columns)[0] == "market"
    pd.testing.assert_frame_equal(res["wf"].drop(columns="market"), want)


def test_study_market_short_history_gives_empty_walk_forward(base_panels, monkeypatch, capsys):
    monkeypatch.setattr(sr, "START", "2020-06-05")                  # ~71 formation dates: < 124
    monkeypatch.setitem(sr.UNIVERSE_N, "US", N_NAMES)
    monkeypatch.setattr(sr, "load_market", lambda root, mkt: {k: v.copy() for k, v in base_panels.items()})
    res = sr.study_market("US", Path("/nowhere"))
    assert res["wf"].empty and "market" not in res["wf"].columns
    out = capsys.readouterr().out
    assert out.startswith(f"  US: {res['n_dates']} dates, ") and out.endswith("s\n")


def test_study_market_uses_market_specific_cost_and_falls_back_to_0_3_percent(base_panels, monkeypatch):
    monkeypatch.setattr(sr, "START", "2020-09-04")
    monkeypatch.setattr(sr, "load_market", lambda root, mkt: {k: v.copy() for k, v in base_panels.items()})
    for mkt, bps in (("IN", 30.0), ("EU", 40.0), ("JP", 20.0)):
        monkeypatch.setitem(sr.UNIVERSE_N, mkt, N_NAMES)
        res = sr.study_market(mkt, Path("/x"))
        assert {round(r["cost_bps"], 9) for r in res["cost"]} == {bps}, mkt
    monkeypatch.setitem(sr.UNIVERSE_N, "KR", N_NAMES)
    monkeypatch.setitem(sr.MARKET_NAME, "KR", "Korea")
    assert {round(r["cost_bps"], 9) for r in sr.study_market("KR", Path("/x"))["cost"]} == {25.0}
    monkeypatch.setitem(sr.UNIVERSE_N, "XX", N_NAMES)
    monkeypatch.setitem(sr.MARKET_NAME, "XX", "Atlantis")           # not in COSTS_PCT
    assert {round(r["cost_bps"], 9) for r in sr.study_market("XX", Path("/x"))["cost"]} == {30.0}


# ── write_markdown ───────────────────────────────────────────────────────────
def ic_row(market, signal, h, mean_ic=0.02, nw_t=1.0, p=0.5, q_bh=0.5, hlz=False, neutral_ic=0.01,
           neutral_t=0.5, bull_ic=0.03, bull_t=1.0, bear_ic=-0.01, bear_t=-0.5):
    return dict(market=market, signal=signal, h=h, mean_ic=mean_ic, nw_t=nw_t, p=p, q_bh=q_bh,
                hlz_t3=hlz, neutral_ic=neutral_ic, neutral_t=neutral_t, bull_ic=bull_ic,
                bull_t=bull_t, bear_ic=bear_ic, bear_t=bear_t)


def md_inputs():
    rows = []
    for m in ("US", "IN"):
        for sg in ("rev5", "mom21", "trend"):
            for h in sr.HORIZONS:
                rows.append(ic_row(m, sg, h))
    ic = pd.DataFrame(rows)
    def setrow(m, sg, h=10, **kw):
        i = ic[(ic.market == m) & (ic.signal == sg) & (ic.h == h)].index[0]
        for k, v in kw.items():
            ic.loc[i, k] = v
    setrow("US", "rev5", bull_ic=-0.02, bull_t=-2.5, bear_ic=0.01, bear_t=0.4)       # WRONG-WAY (bull)
    setrow("US", "mom21", bull_ic=0.04, bull_t=3.0, bear_ic=0.05, bear_t=2.0, q_bh=0.01, hlz_t3=True,
           nw_t=3.4)                                                                  # OK (bear, |t| == 2)
    setrow("IN", "trend", bull_ic=0.01, bull_t=1.99, bear_ic=0.0, bear_t=0.0)         # WEAK (bull)
    setrow("IN", "rev5", bull_ic=0.01, bull_t=2.0, bear_ic=-0.02, bear_t=-2.7)        # strongest, long low values
    zone = [ic_row("US", "zone", h, q_bh=0.001 if h == 10 else 0.5, hlz=(h == 10), nw_t=9 if h == 10 else 1.0)
            for h in sr.HORIZONS]
    ic = pd.concat([ic, pd.DataFrame(zone)], ignore_index=True)
    fm = pd.DataFrame([
        {"market": "US", "signal": "rev5", "fm_coef": 0.01234, "fm_t": 2.34},
        {"market": "US", "signal": "mom21", "fm_coef": -0.002, "fm_t": -0.5},
        {"market": "IN", "signal": "rev5", "fm_coef": 0.0, "fm_t": 0.0}])
    wf = pd.DataFrame([
        dict(market="US", fold=1, test_start="2020-01-03", test_end="2020-06-05", best_single_name="rev5",
             oos_ic_best_single=0.01, oos_ic_ic_weighted=0.02, oos_ic_ridge=0.03),
        dict(market="US", fold=2, test_start="2020-06-12", test_end="2020-12-04", best_single_name="mom21",
             oos_ic_best_single=0.03, oos_ic_ic_weighted=0.04, oos_ic_ridge=0.05)])
    base = dict(h=5, direction="+", ls_spread_pct=1.0, turnover_ls=0.5, breakeven_bps_ls=100.0, cost_bps=10.0,
                ls_net_ann_pct=5.0, long_net_ann_pct=2.0, dsr=0.97)
    cost = pd.DataFrame([
        dict(base, market="US", signal="rev5", ls_net_ann_pct=12.3, long_net_ann_pct=-1.0),
        dict(base, market="US", signal="mom21", ls_net_ann_pct=-3.0, long_net_ann_pct=4.0, dsr=0.5),
        dict(base, market="IN", signal="trend", ls_net_ann_pct=-2.0, long_net_ann_pct=-2.0),     # omitted
        dict(base, market="IN", signal="rev5", ls_net_ann_pct=0.0, long_net_ann_pct=0.0),         # omitted (<= 0)
        dict(base, market="IN", signal="mom21", ls_net_ann_pct=20.0, long_net_ann_pct=-9.0)])
    health = {s: {"status": "HEALTHY", "recent_ic": 0.0123} for s in ("rev5", "mom21", "trend")}
    results = {"US": {"n_dates": 126, "universe": 750, "regime_share_bear": 0.254, "current_regime": "bull",
                      "health": dict(health, rev5={"status": "FLIPPED", "recent_ic": -0.05})},
               "IN": {"n_dates": 100, "universe": 500, "regime_share_bear": 0.5, "current_regime": "bear",
                      "health": health}}
    rules = {"zone_regime.json": {"US": {"bull_rule": "revert", "bear_rule": "mom_st"},
                                  "IN": {"bull_rule": "trend", "bear_rule": "unknown_rule"}}}
    return ic, fm, wf, cost, results, rules


@pytest.fixture
def md(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "HERE", tmp_path)
    ic, fm, wf, cost, results, rules = md_inputs()
    sr.write_markdown(ic, fm, wf, cost, results, rules)
    text = (tmp_path / "SIGNAL_RESEARCH_RESULTS.md").read_text()
    return text, text.splitlines()


def test_markdown_header_and_universe_table(md):
    text, lines = md
    assert lines[0] == "# Signal research results — mailer signals under desk-grade validation"
    assert "| US | 126 | top 750 by 63d median $-volume | 25% | bull |" in lines
    assert "| IN | 100 | top 500 by 63d median $-volume | 50% | bear |" in lines


def test_markdown_survivor_counts_exclude_the_zone_row(md):
    text, _ = md
    # 28 tests (24 + 4 zone rows) have a p-value; the zone row also survives/clears but is not counted
    assert ("28 (market x signal x horizon) tests. **1 survive BH-FDR q<0.05**, "
            "1 clear the Harvey-Liu-Zhu |t|>=3 hurdle.") in text


def test_markdown_ic_table_bolds_survivors_and_formats_cells(md):
    _, lines = md
    assert ("| mom21 | 0.020 (1.0) | **0.020 (3.4)** | 0.020 (1.0) | 0.020 (1.0) | "
            "0.010 (0.5) | 0.040 (3.0) | 0.050 (2.0) |") in lines
    assert ("| rev5 | 0.020 (1.0) | 0.020 (1.0) | 0.020 (1.0) | 0.020 (1.0) | "
            "0.010 (0.5) | -0.020 (-2.5) | 0.010 (0.4) |") in lines
    assert "### US" in lines and "### IN" in lines


def test_markdown_mailer_verdicts(md):
    _, lines = md
    wrong = next(x for x in lines if x.startswith("| US | bull"))
    assert wrong.startswith("| US | bull ◀ now | revert | rev5 | -0.020 (-2.5) | WRONG-WAY | FLIPPED | ")
    ok = next(x for x in lines if x.startswith("| US | bear"))
    assert ok.startswith("| US | bear | mom_st | mom21 | 0.050 (2.0) | OK | HEALTHY | ")     # |t| == 2 is OK
    weak = next(x for x in lines if x.startswith("| IN | bull"))
    assert weak.startswith("| IN | bull | trend | trend | 0.010 (2.0) | WEAK | HEALTHY | ")   # |t| = 1.99
    na = next(x for x in lines if x.startswith("| IN | bear"))
    assert na.startswith("| IN | bear ◀ now | unknown_rule | n/a | n/a | n/a | n/a | ")


def test_markdown_strongest_signal_in_regime_picks_max_abs_t_and_direction(md):
    _, lines = md
    row = lambda pre: next(x for x in lines if x.startswith(pre))
    assert row("| US | bull").endswith("| mom21 0.040 (t 3.0, long high values) |")      # |t| 3.0 beats rev5 -2.5
    assert row("| US | bear").endswith("| mom21 0.050 (t 2.0, long high values) |")
    assert row("| IN | bull").endswith("| rev5 0.010 (t 2.0, long high values) |")        # trend's 1.99 is below the bar
    assert row("| IN | bear").endswith("| rev5 -0.020 (t -2.7, long low values) |")


def test_markdown_strongest_signal_none_when_nothing_reaches_t2(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "HERE", tmp_path)
    ic, fm, wf, cost, results, rules = md_inputs()
    ic["bull_t"], ic["bear_t"] = 1.99, -1.99
    sr.write_markdown(ic, fm, wf, cost, results, rules)
    lines = (tmp_path / "SIGNAL_RESEARCH_RESULTS.md").read_text().splitlines()
    for pre in ("| US | bull", "| US | bear", "| IN | bull", "| IN | bear"):
        assert next(x for x in lines if x.startswith(pre)).endswith("| none with |t|>=2 |")


def test_markdown_fama_macbeth_cells_blank_when_missing(md):
    _, lines = md
    assert "| rev5 | 0.0123 (2.3) | 0.0000 (0.0) |" in lines
    assert "| mom21 | -0.0020 (-0.5) |  |" in lines


def test_markdown_cost_table_omits_loss_makers_and_counts_them(md):
    text, lines = md
    start = lines.index("| market | signal | h | dir | LS spread/period % | turnover | break-even bps | "
                        "cost bps | LS net ann % | long-only net ann % | DSR |")
    rows = []
    for x in lines[start + 2:]:
        if not x.startswith("| "):
            break
        rows.append(x)
    # IN/trend (both negative) and IN/rev5 (0.0 and 0.0, not > 0) are dropped; sorted by LS net, best first
    assert [r.split("|")[1].strip() + "/" + r.split("|")[2].strip() for r in rows] == \
        ["IN/mom21", "US/rev5", "US/mom21"]
    assert "| US | rev5 | 5 | + | 1.00 | 0.50 | 100 | 10 | 12.3 | -1.0 | 0.97 |" in lines
    assert "| US | mom21 | 5 | + | 1.00 | 0.50 | 100 | 10 | -3.0 | 4.0 | 0.50 |" in lines     # long-only > 0 keeps it
    assert "(2 of 5)" in text and "the best of 5 zero-skill trials" in text


def test_markdown_walk_forward_rows_and_fold_means(md):
    _, lines = md
    assert "| US | 1 | 2020-01-03..2020-06-05 | rev5 | 0.010 | 0.020 | 0.030 |" in lines
    assert "| US | 0.020 | 0.030 | 0.040 |" in lines
    assert "Mean across folds:" in lines
    assert not any(x.startswith("| IN | 0.") for x in lines)


def test_markdown_health_table(md):
    _, lines = md
    assert "| rev5 | FLIPPED (-0.050) | HEALTHY (0.012) |" in lines


def test_markdown_without_walk_forward_omits_mean_table(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "HERE", tmp_path)
    ic, fm, wf, cost, results, rules = md_inputs()
    sr.write_markdown(ic, fm, wf.iloc[0:0], cost, results, rules)
    text = (tmp_path / "SIGNAL_RESEARCH_RESULTS.md").read_text()
    assert "Mean across folds:" not in text
    assert "## 7. Live signal health" in text


def test_markdown_unknown_rule_file_defaults_to_na(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "HERE", tmp_path)
    ic, fm, wf, cost, results, rules = md_inputs()
    sr.write_markdown(ic, fm, wf, cost, results, {})
    lines = (tmp_path / "SIGNAL_RESEARCH_RESULTS.md").read_text().splitlines()
    row = next(x for x in lines if x.startswith("| US | bull"))
    assert "| n/a | n/a | n/a | n/a | n/a |" in row


# ── main() end to end (fake warehouse, tmp outputs) ──────────────────────────
@pytest.fixture
def run_main(tmp_path, monkeypatch, base_panels, capsys):
    out = tmp_path / "reports"
    monkeypatch.setattr(sr, "OUT", out)
    monkeypatch.setattr(sr, "HERE", tmp_path)
    monkeypatch.setattr(sr, "START", START)
    monkeypatch.setattr(sr, "MARKETS", ("US",))
    monkeypatch.setitem(sr.UNIVERSE_N, "US", N_NAMES)
    monkeypatch.setattr(sr, "load_market", lambda root, mkt: {k: v.copy() for k, v in base_panels.items()})
    monkeypatch.setenv("HOME", str(tmp_path))

    def go(*argv):
        monkeypatch.setattr(sys, "argv", ["signal_research.py", *argv])
        sr.main()
        return capsys.readouterr().out
    return go, out, tmp_path


def test_main_writes_every_report(run_main):
    go, out, tmp = run_main
    stdout = go("--ohlcv", "/my/ohlcv")
    lines = stdout.splitlines()
    assert lines[0] == "OHLCV root: /my/ohlcv"
    assert lines[-1] == f"wrote {out}/ and SIGNAL_RESEARCH_RESULTS.md"
    for f in ("ic_table.csv", "fama_macbeth_10d.csv", "walk_forward_10d.csv", "costs.csv",
              "signal_health.json"):
        assert (out / f).stat().st_size > 10, f
    assert (tmp / "SIGNAL_RESEARCH_RESULTS.md").read_text().startswith("# Signal research results")

    ic = pd.read_csv(out / "ic_table.csv")
    assert len(ic) == 48
    assert np.allclose(ic["q_bh"], sv.bh_fdr(ic["p"]), equal_nan=True)
    assert (ic["hlz_t3"] == (ic["nw_t"].abs() >= 3.0)).all()
    cost = pd.read_csv(out / "costs.csv")
    assert "_ls_series" not in cost.columns and "dsr" in cost.columns and len(cost) == 33
    assert cost["dsr"].between(0, 1).sum() >= 1

    health = json.loads((out / "signal_health.json").read_text())
    assert list(health) == ["US"]
    assert set(health["US"]) == {"asof", "current_regime", "signals"}
    assert health["US"]["asof"] == str(fridays_from(START)[-1].date())
    assert set(health["US"]["signals"]) == set(SIGNALS)
    assert health["US"]["current_regime"] in {"bull", "bear"}


def test_main_markets_argument_and_default(run_main, monkeypatch, base_panels):
    go, out, tmp = run_main
    seen = []
    def fake_study(m, root):
        seen.append((m, root))
        raise RuntimeError("stop")
    monkeypatch.setattr(sr, "study_market", fake_study)
    with pytest.raises(RuntimeError):
        go("--markets", "IN", "JP", "--ohlcv", "/o")
    assert seen == [("IN", Path("/o"))]
    seen.clear()
    with pytest.raises(RuntimeError):
        go("--ohlcv", "/o")
    assert seen == [("US", Path("/o"))]                      # default: every market in MARKETS
    assert out.is_dir()                                        # OUT is created before the studies run


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="main() concatenates the per-market walk-forward frames without guarding the "
                          "empty case (signal_research.py: `wf = pd.concat([...])`), so a run where no market "
                          "has enough history for a fold dies with 'No objects to concatenate'")
def test_main_survives_runs_where_no_market_has_a_walk_forward_fold(run_main, monkeypatch):
    go, out, tmp = run_main
    monkeypatch.setattr(sr, "START", "2020-09-04")           # too few formation dates for any fold
    stdout = go("--ohlcv", "/o")
    assert stdout.splitlines()[-1] == f"wrote {out}/ and SIGNAL_RESEARCH_RESULTS.md"
