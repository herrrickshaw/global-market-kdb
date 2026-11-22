"""Offline tests for python/extract_sample.py against a tiny Parquet warehouse in tmp_path."""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import extract_sample as ex  # noqa: E402


def write_year(root, mkt, year, rows):
    """rows: (date, symbol, open, high, low, close, volume); prices stored as float32 like the warehouse."""
    d = Path(root) / mkt
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    con = duckdb.connect()
    con.register("df", df)
    con.execute("COPY (SELECT Date, Symbol, Open::FLOAT AS Open, High::FLOAT AS High, Low::FLOAT AS Low, "
                "Close::FLOAT AS Close, Volume FROM df) TO '%s' (FORMAT PARQUET)" % (d / f"year={year}.parquet"))


def bars(sym, close, volume, dates):
    return [(d, sym, close, close * 1.1, close * 0.9, close, volume) for d in dates]


DATES = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]


@pytest.fixture
def wh(tmp_path):
    rows = (bars("BIG", 100.0, 1000, DATES) + bars("MID", 50.0, 100, DATES)
            + bars("SMALL", 10.0, 10, DATES) + bars("TINY", 1.0, 1, DATES))
    write_year(tmp_path, "US", 2020, rows)
    return tmp_path


def test_query_keeps_the_most_liquid_names_sorted_by_date_then_symbol(wh):
    df = ex.query(str(wh), "US", 2, "2020-01-01", "2020-12-31")
    assert list(df.columns) == ["date", "mkt", "sym", "open", "high", "low", "close", "volume"]
    assert sorted(df.sym.unique()) == ["BIG", "MID"]
    assert len(df) == 8
    assert df[["date", "sym"]].apply(tuple, axis=1).is_monotonic_increasing
    assert (df.mkt == "US").all()
    assert df.dtypes["volume"] == "float64" and df.dtypes["close"] == "float64"
    assert pd.Timestamp(df.date.iloc[0]) == pd.Timestamp("2020-01-02")


def test_query_names_argument_is_a_hard_limit(wh):
    assert ex.query(str(wh), "US", 1, "2020-01-01", "2020-12-31").sym.unique().tolist() == ["BIG"]
    assert len(ex.query(str(wh), "US", 4, "2020-01-01", "2020-12-31").sym.unique()) == 4
    assert len(ex.query(str(wh), "US", 99, "2020-01-01", "2020-12-31").sym.unique()) == 4


def test_query_window_is_inclusive_on_both_ends(wh):
    df = ex.query(str(wh), "US", 4, "2020-01-03", "2020-01-06")
    assert sorted(set(pd.to_datetime(df.date).dt.strftime("%Y-%m-%d"))) == ["2020-01-03", "2020-01-06"]
    assert ex.query(str(wh), "US", 4, "2021-01-01", "2021-12-31").empty


def test_query_prices_are_exact_float64_copies_of_float32_source(wh):
    write_year(wh, "JP", 2020, bars("A", 0.1, 5, DATES))
    df = ex.query(str(wh), "JP", 5, "2020-01-01", "2020-12-31")
    assert df.close.iloc[0] == float(np.float32(0.1)) and df.close.iloc[0] != 0.1
    assert df.high.iloc[0] == float(np.float32(0.1 * 1.1))


def test_query_ranks_on_window_dollar_volume_not_share_volume(tmp_path):
    rows = bars("CHEAP", 1.0, 1_000_000, DATES) + bars("PRICEY", 500.0, 10_000, DATES)   # $1m vs $5m
    write_year(tmp_path, "US", 2020, rows)
    assert ex.query(str(tmp_path), "US", 1, "2020-01-01", "2020-12-31").sym.tolist() == ["PRICEY"] * 4


def test_query_zero_volume_days_rank_out_but_stay_in_the_output(tmp_path):
    a = bars("A", 10.0, 100, DATES)
    a[1] = ("2020-01-03", "A", 10.0, 11.0, 9.0, 10.0, 0)
    z = bars("Z", 10.0, 0, DATES)                                    # never trades
    write_year(tmp_path, "US", 2020, a + z)
    df = ex.query(str(tmp_path), "US", 5, "2020-01-01", "2020-12-31")
    assert df.sym.unique().tolist() == ["A"]
    assert (df.volume == 0).sum() == 1 and len(df) == 4


def test_query_median_resists_one_huge_day(tmp_path):
    spike = bars("SPIKE", 10.0, 10, DATES)
    spike[0] = ("2020-01-02", "SPIKE", 10.0, 11.0, 9.0, 10.0, 10_000_000)
    steady = bars("STEADY", 10.0, 500, DATES)
    write_year(tmp_path, "US", 2020, spike + steady)
    assert ex.query(str(tmp_path), "US", 1, "2020-01-01", "2020-12-31").sym.unique().tolist() == ["STEADY"]


def test_query_drops_nonpositive_closes(tmp_path):
    rows = bars("A", 10.0, 100, DATES)
    rows[0] = ("2020-01-02", "A", 10.0, 10.0, 10.0, 0.0, 100)
    rows[1] = ("2020-01-03", "A", 10.0, 10.0, 10.0, -3.0, 100)
    write_year(tmp_path, "US", 2020, rows)
    assert len(ex.query(str(tmp_path), "US", 5, "2020-01-01", "2020-12-31")) == 2


def test_query_collapses_duplicate_symbol_dates(tmp_path):
    rows = bars("A", 10.0, 100, DATES) + bars("A", 10.0, 100, DATES[:2])
    write_year(tmp_path, "US", 2020, rows)
    df = ex.query(str(tmp_path), "US", 5, "2020-01-01", "2020-12-31")
    assert len(df) == 4 and not df.duplicated(["date", "sym"]).any()


def test_query_excludes_indian_etfs_only_for_in(tmp_path):
    syms = ["TCS", "GOLDBEES", "niftybees", "LIQUIDETF", "GILT10", "SENSEXIETF"]
    rows = sum((bars(s, 10.0, 100, DATES[:2]) for s in syms), [])
    write_year(tmp_path, "IN", 2020, rows)
    write_year(tmp_path, "US", 2020, rows)
    assert ex.query(str(tmp_path), "IN", 10, "2020-01-01", "2020-12-31").sym.unique().tolist() == ["TCS"]
    assert len(ex.query(str(tmp_path), "US", 10, "2020-01-01", "2020-12-31").sym.unique()) == 6


@pytest.mark.parametrize("start,end", [("2020-1-1", "2020-12-31"), ("2020-01-01", "20201231"),
                                       ("2020-01-01'; DROP TABLE x; --", "2020-12-31"),
                                       ("2020-01-01", "2020-12-31 "), ("", "2020-12-31"),
                                       ("2020-01-01", "x2020-12-31")])
def test_query_rejects_malformed_dates_before_touching_the_files(tmp_path, start, end):
    with pytest.raises(ValueError, match="dates must be YYYY-MM-DD"):
        ex.query(str(tmp_path / "nonexistent"), "US", 5, start, end)


def test_main_writes_csv_and_reports(wh, tmp_path, capsys, monkeypatch):
    out = tmp_path / "sub" / "dir" / "us.csv"
    monkeypatch.setattr(sys, "argv", ["extract_sample.py", "--root", str(wh), "--mkt", "US", "--names", "2",
                                      "--from", "2020-01-01", "--to", "2020-12-31", "--out", str(out)])
    ex.main()
    assert capsys.readouterr().out == f"wrote {out}: 8 rows, 2 names\n"
    back = pd.read_csv(out)
    assert list(back.columns) == ["date", "mkt", "sym", "open", "high", "low", "close", "volume"]
    assert back.date.iloc[0] == "2020-01-02" and len(back) == 8
    assert back.close.iloc[0] == float(np.float32(100.0))


def test_main_defaults_come_from_home_and_data_local(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    root = home / ".cache" / "market-warehouse" / "ohlcv"
    write_year(root, "US", 2018, bars("A", 10.0, 100, ["2018-03-01", "2018-03-02"]))
    write_year(root, "US", 2021, bars("A", 10.0, 100, ["2021-03-01"]))             # after the default window
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(ex, "ROOT", tmp_path / "repo")
    monkeypatch.setattr(sys, "argv", ["extract_sample.py"])
    ex.main()
    target = tmp_path / "repo" / "data" / "local" / "US.csv"
    assert capsys.readouterr().out == f"wrote {target}: 2 rows, 1 names\n"
    assert len(pd.read_csv(target)) == 2


def test_main_rejects_unknown_market(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["extract_sample.py", "--mkt", "XX"])
    with pytest.raises(SystemExit) as e:
        ex.main()
    assert e.value.code == 2


def test_main_names_default_is_150(tmp_path, monkeypatch, capsys):
    rows = sum((bars(f"S{i:03d}", 10.0, 100 + i, ["2018-03-01"]) for i in range(160)), [])
    write_year(tmp_path / "wh", "US", 2018, rows)
    out = tmp_path / "o.csv"
    monkeypatch.setattr(sys, "argv", ["extract_sample.py", "--root", str(tmp_path / "wh"), "--out", str(out)])
    ex.main()
    assert capsys.readouterr().out == f"wrote {out}: 150 rows, 150 names\n"
