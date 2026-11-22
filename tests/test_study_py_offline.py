"""Offline tests for python/study_py.py (the Python parity oracle runner) on synthetic CSVs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import study_py  # noqa: E402

sr, sv = study_py.sr, study_py.sv


def interchange(markets, n_days=1000, n=70, seed=0, start="2018-01-01"):
    frames = []
    for j, (mkt, days_cut) in enumerate(markets.items()):
        rng = np.random.default_rng(seed + j)
        days = pd.bdate_range(start, periods=days_cut or n_days)
        syms = [f"{mkt[0]}{i:02d}" for i in range(n)]
        r = rng.normal(0.0003, 0.015, (len(days), n))
        close = 40 * np.exp(np.cumsum(r, axis=0))
        df = pd.DataFrame({
            "date": np.repeat(days.values, n), "mkt": mkt, "sym": np.tile(syms, len(days)),
            "open": close.ravel(), "high": (close * 1.01).ravel(), "low": (close * 0.99).ravel(),
            "close": close.ravel(), "volume": np.round(rng.lognormal(12, 0.4, close.shape)).ravel()})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def isolated_sr(monkeypatch):
    """study_py.run() rewires signal_research globals; restore them afterwards."""
    monkeypatch.setattr(sr, "START", sr.START)
    monkeypatch.setattr(sr, "load_market", sr.load_market)
    monkeypatch.setattr(sr, "UNIVERSE_N", dict(sr.UNIVERSE_N))
    monkeypatch.setattr(sr, "MARKET_NAME", dict(sr.MARKET_NAME))


# ── load_panels ──────────────────────────────────────────────────────────────
def small_frame():
    return pd.DataFrame({
        "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03", "2020-01-02", "2020-01-03"],
        "mkt": ["AA", "AA", "AA", "AA", "BB", "BB"],
        "sym": ["x", "y", "x", "y", "z", "z"],
        "open": [1, 1, 1, 1, 1, 1], "low": [1, 1, 1, 1, 1, 1],
        "high": [11.0, 21.0, 12.0, 22.0, 31.0, 32.0],
        "close": [10.0, 20.0, 10.5, 0.0, 30.0, 31.0],
        "volume": [100, 200, 110, 210, 300, 310]})


def test_load_panels_pivots_each_market_and_drops_nonpositive_closes():
    panels = study_py.load_panels(small_frame())
    assert set(panels) == {"AA", "BB"}
    aa = panels["AA"]
    assert set(aa) == {"Close", "High", "Volume"}
    assert list(aa["Close"].columns) == ["x", "y"]
    assert aa["Close"].loc["2020-01-02", "x"] == 10.0
    # the close == 0 row is dropped from every panel, so y has no 2020-01-03 entry at all
    assert np.isnan(aa["Close"].loc["2020-01-03", "y"]) and np.isnan(aa["High"].loc["2020-01-03", "y"])
    assert np.isnan(aa["Volume"].loc["2020-01-03", "y"])
    assert aa["High"].loc["2020-01-03", "x"] == 12.0 and aa["Volume"].loc["2020-01-02", "y"] == 200
    for p in panels.values():
        for df in p.values():
            assert df.dtypes.eq("float64").all()
            assert isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing
    assert panels["BB"]["Close"].shape == (2, 1)


def test_load_panels_negative_close_dropped_and_last_duplicate_wins():
    df = small_frame()
    df.loc[0, "close"] = -1.0
    dup = df.iloc[[2]].assign(close=99.0, high=98.0)           # same (date, sym) later in the file
    out = study_py.load_panels(pd.concat([df, dup], ignore_index=True))
    assert np.isnan(out["AA"]["Close"].loc["2020-01-02", "x"])
    assert out["AA"]["Close"].loc["2020-01-03", "x"] == 99.0   # aggfunc "last"
    assert out["AA"]["High"].loc["2020-01-03", "x"] == 98.0


def test_load_panels_reads_csv_paths_and_does_not_mutate_dataframes(tmp_path):
    df = small_frame()
    f = tmp_path / "in.csv"
    df.to_csv(f, index=False)
    from_csv = study_py.load_panels(str(f))
    from_df = study_py.load_panels(df)
    from_path = study_py.load_panels(f)
    for m in from_df:
        for k in from_df[m]:
            pd.testing.assert_frame_equal(from_csv[m][k], from_df[m][k])
            pd.testing.assert_frame_equal(from_path[m][k], from_df[m][k])
    assert df["date"].tolist()[0] == "2020-01-02"               # caller's frame is left as strings


# ── run ──────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def two_market_run():
    mp = pytest.MonkeyPatch()
    for name in ("START", "load_market"):
        mp.setattr(sr, name, getattr(sr, name))
    mp.setattr(sr, "UNIVERSE_N", dict(sr.UNIVERSE_N))
    mp.setattr(sr, "MARKET_NAME", dict(sr.MARKET_NAME))
    try:
        src = interchange({"AAA": None, "BBB": 800})
        res = study_py.run(src, 70, "2019-06-07")
        state = {"START": sr.START, "UNIVERSE_N": dict(sr.UNIVERSE_N), "MARKET_NAME": dict(sr.MARKET_NAME),
                 "panel_copy": sr.load_market(None, "AAA")}
    finally:
        mp.undo()
    return res, state, src


def test_run_returns_the_six_parity_tables(two_market_run):
    res, _, _ = two_market_run
    assert set(res) == {"ic_series", "ic_table", "fama_macbeth_10d", "walk_forward_10d", "costs", "health"}
    assert set(res["ic_series"].columns) == {"market", "signal", "h", "date", "ic"}
    assert set(res["ic_series"].market) == {"AAA", "BBB"}
    assert set(res["ic_series"].h) == {5, 10, 21, 63}
    assert "zone" in set(res["ic_series"].signal)                # the series table keeps every signal incl. zone


def test_run_ic_table_has_cross_market_fdr_and_hlz_flags(two_market_run):
    res, _, _ = two_market_run
    ic = res["ic_table"]
    assert len(ic) == 2 * 12 * 4
    assert np.allclose(ic["q_bh"], sv.bh_fdr(ic["p"]), equal_nan=True)          # one family across markets
    assert (ic["hlz_t3"] == (ic["nw_t"].abs() >= 3.0)).all()
    assert set(ic.market) == {"AAA", "BBB"}


def test_run_costs_use_deflated_sharpe_over_all_markets_and_drop_the_series(two_market_run):
    res, _, _ = two_market_run
    cost = res["costs"]
    assert "_ls_series" not in cost.columns and len(cost) == 2 * 11 * 3
    assert cost["dsr"].dropna().between(0, 1).all() and cost["dsr"].notna().any()
    assert set(cost.cost_bps.round(6)) == {30.0}                # unknown markets fall back to 0.3% round trip
    assert set(cost.market) == {"AAA", "BBB"}


def test_run_health_and_fama_macbeth_and_walk_forward(two_market_run):
    res, _, _ = two_market_run
    health = res["health"]
    assert list(health.columns)[:2] == ["market", "signal"]
    assert len(health) == 22 and set(health.status) <= {"HEALTHY", "DECAYING", "FLIPPED", "NOISE"}
    assert len(res["fama_macbeth_10d"]) == 22
    wf = res["walk_forward_10d"]
    assert set(wf.market) == {"AAA"}                            # BBB is too short for any fold, AAA is kept
    assert len(wf) == 5


def test_run_rewires_signal_research_globals(two_market_run):
    _, state, _ = two_market_run
    assert state["START"] == "2019-06-07"
    assert state["UNIVERSE_N"]["AAA"] == 70 and state["UNIVERSE_N"]["BBB"] == 70
    assert state["MARKET_NAME"]["AAA"] == "AAA"
    assert state["MARKET_NAME"]["US"] == "US"                   # existing names are not overwritten
    p = state["panel_copy"]
    assert set(p) == {"Close", "High", "Volume"} and p["Close"].shape[1] == 70


def test_run_keeps_existing_market_name_for_known_codes(isolated_sr):
    src = interchange({"IN": 800}, n_days=800)
    res = study_py.run(src, 70, "2019-06-07")
    assert set(res["costs"].cost_bps.round(6)) == {30.0}        # India: 0.30% from COSTS_PCT
    assert sr.MARKET_NAME["IN"] == "India"
    assert sr.UNIVERSE_N["IN"] == 70


def test_run_is_silent_and_handles_no_walk_forward_anywhere(isolated_sr, capsys):
    res = study_py.run(interchange({"CCC": 800}), 70, "2019-06-07")
    assert capsys.readouterr().out == ""                         # study_market's progress print is swallowed
    assert isinstance(res["walk_forward_10d"], pd.DataFrame) and res["walk_forward_10d"].empty


def test_run_accepts_csv_path(isolated_sr, tmp_path):
    f = tmp_path / "x.csv"
    interchange({"DDD": 800}).to_csv(f, index=False)
    res = study_py.run(str(f), 70, "2019-06-07")
    assert set(res["ic_table"].market) == {"DDD"}


# ── main ─────────────────────────────────────────────────────────────────────
def test_main_writes_every_table_with_full_precision(tmp_path, monkeypatch, capsys):
    tables = {"ic_series": pd.DataFrame({"date": pd.to_datetime(["2020-01-03"]), "ic": [0.1]}),
              "ic_table": pd.DataFrame({"x": [1.0 / 3.0, 2.0]}),
              "costs": pd.DataFrame({"y": [1, 2, 3]})}
    seen = []
    monkeypatch.setattr(study_py, "run", lambda src, n, start: seen.append((src, n, start)) or tables)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["study_py.py", "in.csv"])
    study_py.main()
    assert seen == [("in.csv", 100, "2019-01-01")]
    out = tmp_path / "out" / "py"
    assert (out / "ic_series.csv").read_text() == "date,ic\n2020-01-03,0.10000000000000001\n"
    assert (out / "ic_table.csv").read_text() == "x\n0.33333333333333331\n2\n"
    assert (out / "costs.csv").read_text() == "y\n1\n2\n3\n"
    assert capsys.readouterr().out.splitlines() == ["ic_series: 1 rows", "ic_table: 2 rows", "costs: 3 rows"]


def test_main_options(tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(study_py, "run", lambda src, n, start: seen.append((src, n, start)) or {"t": pd.DataFrame({"a": [1]})})
    monkeypatch.setattr(sys, "argv", ["study_py.py", "d.csv", "--n", "55", "--start", "2020-03-01",
                                      "--out", str(tmp_path / "a" / "b" / "c")])
    study_py.main()
    assert seen == [("d.csv", 55, "2020-03-01")]
    assert (tmp_path / "a" / "b" / "c" / "t.csv").read_text() == "a\n1\n"
    assert capsys.readouterr().out == "t: 1 rows\n"
