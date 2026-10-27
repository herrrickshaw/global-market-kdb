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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path.home() / ".cache/market-warehouse/ohlcv"))
    ap.add_argument("--mkt", default="US", choices=["IN", "US", "JP", "KR", "EU", "CN"])
    ap.add_argument("--names", type=int, default=150)
    ap.add_argument("--from", dest="start", default="2018-01-01")
    ap.add_argument("--to", dest="end", default="2020-12-31")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.start) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.end):
        ap.error("dates must be YYYY-MM-DD")

    glob = f"{a.root}/{a.mkt}/*.parquet"
    out = Path(a.out or ROOT / "data" / "local" / f"{a.mkt}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    etf = f"AND NOT regexp_matches(upper(Symbol), '{ETF_RE}')" if a.mkt == "IN" else ""
    con = duckdb.connect()
    con.sql(f"""
        WITH w AS (
          SELECT * FROM '{glob}'
          WHERE Date >= DATE '{a.start}' AND Date <= DATE '{a.end}' AND Close > 0 {etf}),
        keep AS (
          SELECT Symbol FROM w WHERE Volume > 0 GROUP BY 1
          ORDER BY median(Close * Volume) DESC LIMIT {int(a.names)})
        SELECT strftime(Date, '%Y-%m-%d') AS date, '{a.mkt}' AS mkt, Symbol AS sym,
               Open AS open, High AS high, Low AS low, Close AS close, Volume AS volume
        FROM w JOIN keep USING (Symbol)
        QUALIFY row_number() OVER (PARTITION BY Symbol, Date ORDER BY Date) = 1
        ORDER BY date, sym
    """).write_csv(str(out))
    n = con.sql(f"SELECT count(*), count(DISTINCT sym) FROM read_csv('{out}')").fetchall()[0]
    print(f"wrote {out}: {n[0]:,} rows, {n[1]} names")


if __name__ == "__main__":
    main()
