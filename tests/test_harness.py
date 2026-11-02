"""Unit tests for the Python parity harness (no q, no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import extract_sample  # noqa: E402
import make_sample  # noqa: E402
import parity  # noqa: E402
import study_py  # noqa: E402


def ic_table(**over):
    row = {"market": "SYN", "signal": "rev5", "h": 5, "mean_ic": 0.02, "nw_t": 1.5}
    row.update(over)
    return pd.DataFrame([row])


def test_compare_identical_tables_pass():
    r = parity.compare("ic_table", ic_table(), ic_table(), 1e-10, 1e-8)
    assert r["pass"] and r["mismatches"] == 0 and r["rows"] == 1


def test_compare_within_and_outside_tolerance():
    ok = parity.compare("ic_table", ic_table(mean_ic=0.02 + 1e-12), ic_table(), 1e-10, 1e-8)
    bad = parity.compare("ic_table", ic_table(mean_ic=0.03), ic_table(), 1e-10, 1e-8)
    assert ok["pass"]
    assert not bad["pass"] and bad["worst"] == "mean_ic"


def test_compare_null_pattern_must_match():
    r = parity.compare("ic_table", ic_table(nw_t=np.nan), ic_table(), 1e-10, 1e-8)
    assert not r["pass"]


def test_compare_missing_key_row_is_a_mismatch():
    q = pd.concat([ic_table(), ic_table(h=10)], ignore_index=True)
    r = parity.compare("ic_table", q, ic_table(), 1e-10, 1e-8)
    assert not r["pass"] and r["mismatches"] >= 1


def test_compare_empty_both_sides_passes_one_side_fails():
    assert parity.compare("health", None, pd.DataFrame(), 1e-10, 1e-8)["pass"]
    assert not parity.compare("health", ic_table(), pd.DataFrame(), 1e-10, 1e-8)["pass"]


def test_compare_exempts_lowvol_neutral_values_but_not_nulls():
    py = ic_table(signal="lowvol", neutral_ic=1e-3)
    wild = ic_table(signal="lowvol", neutral_ic=0.9)
    assert parity.compare("ic_table", wild, py, 1e-10, 1e-8)["pass"]
    nulled = ic_table(signal="lowvol", neutral_ic=np.nan)
    assert not parity.compare("ic_table", nulled, py, 1e-10, 1e-8)["pass"]


def test_norm_parses_dates_and_bools():
    df = parity._norm(pd.DataFrame({"d": ["2020.01.02", "2020.01.03"], "f": ["True", "False"], "x": [1, 2]}))
    assert str(df["d"].dtype).startswith("datetime64")
    assert df["f"].tolist() == [True, False]


def test_q_commands_build_csv_and_parquet_invocations():
    hdb, out = Path("h"), Path("o")
    load, run = parity._q_commands(Path("a.csv"), 100, "2019-01-01", hdb, out, None)
    assert load == ["q/load.q", "-csv", "a.csv", "-hdb", "h"]
    assert run[-2:] == ["-out", "o"] and "2019.01.01" in run
    pq = {"root": "/w", "mkts": ["US", "IN"], "from": "2016-06-01", "to": "2022-12-31", "names": 5}
    load, _ = parity._q_commands(None, 10, "2017-07-01", hdb, out, pq)
    assert load[:3] == ["q/pqload.q", "-root", "/w"] and "2016.06.01" in load and "US" in load


def test_find_q_honours_env(monkeypatch, tmp_path):
    fake = tmp_path / "q"
    fake.write_text("")
    monkeypatch.setenv("Q", str(fake))
    assert parity.find_q() == str(fake)


def test_make_sample_regenerates_the_committed_csv_byte_for_byte(tmp_path, monkeypatch):
    out = tmp_path / "ohlcv.csv"
    monkeypatch.setattr(make_sample, "ROOT", tmp_path)   # main() prints OUT relative to ROOT
    monkeypatch.setattr(make_sample, "OUT", out)
    make_sample.main()
    assert out.read_bytes() == (ROOT / "data/sample/ohlcv.csv").read_bytes()


def test_load_panels_pivots_per_market_and_drops_nonpositive_close():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"] * 2, "mkt": "M", "sym": ["A", "A", "B", "B"],
                       "open": 1.0, "high": 2.0, "low": 0.5, "close": [1.0, 0.0, 3.0, 4.0],
                       "volume": 10.0})
    p = study_py.load_panels(df)["M"]
    assert set(p) == {"Close", "High", "Volume"}
    assert p["Close"].shape == (2, 2) and np.isnan(p["Close"].loc["2020-01-02", "A"])


def test_extract_query_picks_most_liquid_names_and_validates_dates(tmp_path):
    d = tmp_path / "US"
    d.mkdir()
    rows = [(f"2020-01-0{i}", s, 1.0, 2.0, 0.5, 1.0, v) for s, v in (("BIG", 1e6), ("SMALL", 10.0))
            for i in (1, 2, 3)]
    con = duckdb.connect()
    con.execute("CREATE TABLE w(Date DATE, Symbol VARCHAR, Open FLOAT, High FLOAT, Low FLOAT, "
                "Close FLOAT, Volume DOUBLE)")
    con.executemany("INSERT INTO w VALUES (?,?,?,?,?,?,?)", rows)
    con.execute(f"COPY w TO '{d}/year=2020.parquet' (FORMAT parquet)")
    got = extract_sample.query(str(tmp_path), "US", 1, "2020-01-01", "2020-12-31")
    assert set(got["sym"]) == {"BIG"} and len(got) == 3 and set(got["mkt"]) == {"US"}
    with pytest.raises(ValueError):
        extract_sample.query(str(tmp_path), "US", 1, "2020/01/01", "2020-12-31")
