#!/usr/bin/env python3
"""Extract a real slice of the Parquet warehouse into the interchange CSV.

Reads the hive-style warehouse (<root>/<MKT>/year=YYYY.parquet, columns
Date,Symbol,Open,High,Low,Close,Volume) with DuckDB, keeps the `--names` most
liquid symbols by median dollar volume over the window (the same liquidity
idea as upstream load_market's pre-filter) and writes
date,mkt,sym,open,high,low,close,volume for q/load.q.

Output goes to data/local/ (gitignored): warehouse prices come from market
data vendors and are not redistributed in this repository.

Run:
  python python/extract_sample.py --root ~/.cache/market-warehouse/ohlcv \
      --mkt US --names 150 --from 2018-01-01 --to 2020-12-31
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
ETF_RE = r"BEES|ETF|LIQUID|GILT|NIFTY|SENSEX"    # upstream India ETF exclusion


def query(root: str, mkt: str, names: int, start: str, end: str):
    """The extraction as a DataFrame (prices as exact float64 copies of the float32 source)."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", end):
        raise ValueError("dates must be YYYY-MM-DD")
    glob = f"{root}/{mkt}/*.parquet"
    etf = f"AND NOT regexp_matches(upper(Symbol), '{ETF_RE}')" if mkt == "IN" else ""
    return duckdb.connect().sql(f"""
        WITH w AS (
          SELECT * FROM '{glob}'
          WHERE Date >= DATE '{start}' AND Date <= DATE '{end}' AND Close > 0 {etf}),
        keep AS (
          SELECT Symbol FROM w WHERE Volume > 0 GROUP BY 1
          ORDER BY median(Close::DOUBLE * Volume) DESC LIMIT {int(names)})
        SELECT CAST(Date AS DATE) AS date, '{mkt}' AS mkt, Symbol AS sym,
               Open::DOUBLE AS open, High::DOUBLE AS high, Low::DOUBLE AS low,
               Close::DOUBLE AS close, Volume::DOUBLE AS volume
        FROM w JOIN keep USING (Symbol)
        QUALIFY row_number() OVER (PARTITION BY Symbol, Date ORDER BY Date) = 1
        ORDER BY date, sym
    """).df()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path.home() / ".cache/market-warehouse/ohlcv"))
    ap.add_argument("--mkt", default="US", choices=["IN", "US", "JP", "KR", "EU", "CN"])
    ap.add_argument("--names", type=int, default=150)
    ap.add_argument("--from", dest="start", default="2018-01-01")
    ap.add_argument("--to", dest="end", default="2020-12-31")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out or ROOT / "data" / "local" / f"{a.mkt}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = query(a.root, a.mkt, a.names, a.start, a.end)
    df.to_csv(out, index=False, date_format="%Y-%m-%d")   # repr round-trips each double exactly
    print(f"wrote {out}: {len(df):,} rows, {df['sym'].nunique()} names")


if __name__ == "__main__":
    main()
