"""Offline tests for python/make_sample.py: the seeded synthetic-sample generator."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import make_sample as ms  # noqa: E402


def generate(tmp_path, monkeypatch, capsys, name="a"):
    repo = tmp_path / name
    monkeypatch.setattr(ms, "ROOT", repo)
    monkeypatch.setattr(ms, "OUT", repo / "data" / "sample" / "ohlcv.csv")
    ms.main()
    return ms.OUT, capsys.readouterr().out


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    repo = tmp_path_factory.mktemp("ms")
    mp.setattr(ms, "ROOT", repo)
    mp.setattr(ms, "OUT", repo / "data" / "sample" / "ohlcv.csv")
    import contextlib
    import io
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ms.main()
    finally:
        mp.undo()
    path = repo / "data" / "sample" / "ohlcv.csv"
    return path, pd.read_csv(path, parse_dates=["date"]), buf.getvalue()


def test_constants_describe_the_documented_universe():
    assert ms.N_NAMES == 120 and ms.SEED == 20261027
    assert ms.DATES[0] == pd.Timestamp("2018-01-01") and ms.DATES[-1] == pd.Timestamp("2022-12-30")
    assert (ms.DATES.dayofweek < 5).all() and len(ms.DATES) == 1305


def test_output_schema_and_report_line(sample):
    path, df, out = sample
    assert list(df.columns) == ["date", "mkt", "sym", "open", "high", "low", "close", "volume"]
    assert (df.mkt == "SYN").all()
    assert out == (f"wrote data/sample/ohlcv.csv: {len(df):,} rows, 120 names, "
                   f"{df.date.min():%Y-%m-%d}..{df.date.max():%Y-%m-%d}\n")
    assert sorted(df.sym.unique()) == [f"S{i:03d}" for i in range(1, 121)]
    assert df.date.min() >= ms.DATES[0] and df.date.max() <= ms.DATES[-1]
    assert path.read_text().splitlines()[1].split(",")[0] == df.date.min().strftime("%Y-%m-%d")   # ISO dates in the file


def test_rows_are_sorted_by_date_then_symbol_without_duplicates(sample):
    _, df, _ = sample
    keys = list(zip(df.date, df.sym))
    assert keys == sorted(keys) and len(set(keys)) == len(keys)


def test_about_one_percent_of_rows_are_missing_and_lifecycles_trim_more(sample):
    _, df, _ = sample
    full = len(ms.DATES) * ms.N_NAMES
    assert 0.85 * full < len(df) < 0.995 * full
    per_name = df.groupby("sym").size()
    assert per_name.min() < per_name.max() * 0.9                  # late listings / early delistings exist


def test_missing_rows_inside_each_names_life_are_about_one_percent(sample):
    _, df, _ = sample
    pos = pd.Series(np.arange(len(ms.DATES)), index=ms.DATES)
    g = df.assign(pos=df.date.map(pos)).groupby("sym")["pos"].agg(["min", "max", "size"])
    span = (g["max"] - g["min"] + 1).sum()
    missing = 1 - g["size"].sum() / span
    assert 0.005 < missing < 0.02


def test_prices_are_rounded_positive_and_consistent(sample):
    _, df, _ = sample
    for c in ("open", "high", "low", "close"):
        assert (df[c] > 0).all()
        assert np.allclose(df[c], df[c].round(4))
        assert (df[c] != df[c].round(3)).mean() > 0.5              # four decimals are really there
    assert (df.high >= df.close - 1e-4).all() and (df.low <= df.close + 1e-4).all()
    assert ((df.open >= df.low - 1e-4) & (df.open <= df.high + 1e-4)).all()
    assert df.volume.dtype == np.int64 and (df.volume >= 0).all()


def test_awkward_cases_the_parity_test_relies_on_are_present(sample):
    _, df, _ = sample
    assert 0.005 < (df.volume == 0).mean() < 0.02                  # zero-volume days, about 1% of rows
    assert (df.high == df.close).mean() > 0.05                     # High == Close ties
    ratio = df.sort_values(["sym", "date"]).groupby("sym")["close"].pct_change()
    assert ((ratio < -0.5) | (ratio > 1.0)).sum() >= 3             # unadjusted -60% / +150% corporate-action jumps
    gaps = df.groupby("sym").date.apply(lambda s: (s.diff().dt.days > 3).sum())
    assert gaps.sum() > 100                                        # missing rows create gaps longer than a weekend


def test_return_dynamics_match_the_documented_design(sample):
    _, df, _ = sample
    px = df.pivot(index="date", columns="sym", values="close")
    r = np.log(px / px.shift(1))
    consecutive = px.notna() & px.shift(1).notna()
    r = r.where(consecutive & (r.abs() < 0.3))                         # drop gaps and the injected jumps
    mkt = r.mean(axis=1)
    assert 0.008 < mkt.std() < 0.0135                                  # market factor: 1.0% daily vol, beta ~ 1
    idio = r.sub(mkt, axis=0)
    lag0, lag1 = idio.stack(), idio.shift(1).stack()
    both = pd.concat([lag0, lag1], axis=1, join="inner")
    assert both.corr().iloc[0, 1] < -0.03                               # mild reversal: AR(1) with phi = -0.05


def test_generation_is_deterministic(tmp_path, monkeypatch, capsys):
    p1, _ = generate(tmp_path, monkeypatch, capsys, "one")
    p2, _ = generate(tmp_path, monkeypatch, capsys, "two")
    assert hashlib.sha256(p1.read_bytes()).hexdigest() == hashlib.sha256(p2.read_bytes()).hexdigest()


def test_seed_changes_the_data(tmp_path, monkeypatch, capsys):
    p1, _ = generate(tmp_path, monkeypatch, capsys, "one")
    monkeypatch.setattr(ms, "SEED", 1)
    p2, _ = generate(tmp_path, monkeypatch, capsys, "two")
    assert p1.read_bytes() != p2.read_bytes()


def test_universe_size_is_configurable_and_output_directory_created(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ms, "N_NAMES", 12)
    monkeypatch.setattr(ms, "DATES", pd.bdate_range("2018-01-01", "2019-12-31"))
    path, out = generate(tmp_path, monkeypatch, capsys, "small")
    df = pd.read_csv(path)
    assert df.sym.nunique() == 12 and path.parent.is_dir()
    assert out.startswith("wrote data/sample/ohlcv.csv: ") and ", 12 names, 2018-01-01.." in out


def test_committed_sample_has_the_same_schema(sample):
    _, df, _ = sample
    committed = pd.read_csv(ROOT / "data" / "sample" / "ohlcv.csv", nrows=5)
    assert list(committed.columns) == list(df.columns)
