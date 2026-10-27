#!/usr/bin/env python3
"""The DuckDB and Polars side of bench/run.py: the same five operations as
bench/bench.q, one JSON line per (engine, op). Threads are fixed with --threads."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


MIN = {}


def tm(f, reps):
    out, v = [], None
    for _ in range(reps):
        s = time.perf_counter()
        v = f()
        out.append((time.perf_counter() - s) * 1e3)
    MIN["last"] = min(out)
    return statistics.median(out), v


def emit(engine, op, ms, chk, threads):
    print(json.dumps({"engine": engine, "op": op, "ms": ms, "min_ms": MIN["last"], "check": chk,
                      "threads": threads}), flush=True)


def run_duckdb(root, reps, threads):
    import duckdb
    con = duckdb.connect()

    def mat(sql, check):
        """Materialise sql as a temp table (so the work is not deferred), return a check value."""
        con.execute(f"CREATE OR REPLACE TEMP TABLE out AS {sql}")
        return con.execute(f"SELECT {check} FROM out").fetchone()[0]
    con.execute(f"SET threads={threads}")
    us, ind = f"{root}/ohlcv/US/*.parquet", f"{root}/ohlcv/IN/*.parquet"
    fund = f"{root}/fundamentals_history/IN_screener_only_backup.parquet"
    eng = f"duckdb-{threads}t"

    def load():
        con.execute(f"CREATE OR REPLACE TABLE t AS SELECT Date::DATE AS date, Symbol AS sym, "
                    f"Close::DOUBLE AS close, Volume AS volume FROM '{us}'")
        return con.execute("SELECT count(*) FROM t").fetchone()[0]
    ms, n = tm(load, reps); emit(eng, "load_us", ms, n, threads)

    q2 = ("SELECT sym, year(date) y, median(close*volume) m FROM t WHERE close>0 AND volume>0 "
          "GROUP BY 1, 2")
    ms, n = tm(lambda: mat(q2, "count(*)"), reps)
    emit(eng, "groupby_median", ms, n, threads)

    q3 = ("SELECT *, close/lag(close,5) OVER w - 1 AS rev5, "
          "avg(abs(r1)) OVER (PARTITION BY sym ORDER BY date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW) AS am "
          "FROM (SELECT *, close/lag(close) OVER w - 1 AS r1 FROM t WINDOW w AS (PARTITION BY sym ORDER BY date)) "
          "WINDOW w AS (PARTITION BY sym ORDER BY date)")
    ms, s = tm(lambda: mat(q3, "sum(am)"), reps)
    con.execute(f"CREATE OR REPLACE TABLE x AS {q3}")
    emit(eng, "window_by_sym", ms, s, threads)

    q4 = "SELECT *, row_number() OVER (PARTITION BY date ORDER BY rev5) AS rk FROM x"
    ms, n = tm(lambda: mat(q4, "count(*)"), reps)
    emit(eng, "rank_by_date", ms, n, threads)

    con.execute(f"CREATE OR REPLACE TABLE ind AS SELECT Date::DATE AS date, Symbol AS sym, Close::DOUBLE AS close FROM '{ind}'")
    con.execute(f"CREATE OR REPLACE TABLE f AS SELECT ticker AS sym, fy_end + 90 AS date, net_income AS ni, shares FROM '{fund}'")
    q5 = "SELECT ind.*, f.ni, f.shares FROM ind ASOF LEFT JOIN f ON ind.sym = f.sym AND ind.date >= f.date"
    ms, n = tm(lambda: mat(q5, "count(ni)"), reps)
    emit(eng, "asof_join_in", ms, n, threads)


def run_polars(root, reps, threads):
    import polars as pl
    eng = f"polars-{threads}t"
    us = sorted(Path(f"{root}/ohlcv/US").glob("year=*.parquet"))

    def load():   # year files mix float32/float64 prices, so cast each before stacking
        return pl.concat([pl.read_parquet(f, columns=["Date", "Symbol", "Close", "Volume"]).select(
            pl.col("Date").cast(pl.Date).alias("date"), pl.col("Symbol").alias("sym"),
            pl.col("Close").cast(pl.Float64).alias("close"), pl.col("Volume").cast(pl.Float64).alias("volume"))
            for f in us])
    ms, t = tm(load, reps); emit(eng, "load_us", ms, t.height, threads)

    ms, g = tm(lambda: t.filter((pl.col("close") > 0) & (pl.col("volume") > 0))
               .group_by("sym", pl.col("date").dt.year().alias("y"))
               .agg((pl.col("close") * pl.col("volume")).median().alias("m")), reps)
    emit(eng, "groupby_median", ms, g.height, threads)

    t = t.sort("sym", "date")
    ms, x = tm(lambda: t.with_columns(
        (pl.col("close") / pl.col("close").shift(5) - 1).over("sym").alias("rev5"),
        (pl.col("close") / pl.col("close").shift(1) - 1).abs().rolling_mean(63, min_samples=1).over("sym").alias("am")), reps)
    emit(eng, "window_by_sym", ms, x["am"].sum(), threads)

    ms, k = tm(lambda: x.with_columns(pl.col("rev5").rank("ordinal").over("date").alias("rk")), reps)
    emit(eng, "rank_by_date", ms, k.height, threads)

    ind = pl.concat([pl.read_parquet(f, columns=["Date", "Symbol", "Close"]).select(
        pl.col("Date").cast(pl.Date).alias("date"), pl.col("Symbol").alias("sym"),
        pl.col("Close").cast(pl.Float64).alias("close"))
        for f in sorted(Path(f"{root}/ohlcv/IN").glob("year=*.parquet"))]).sort("date")
    f = pl.read_parquet(f"{root}/fundamentals_history/IN_screener_only_backup.parquet").select(
        pl.col("ticker").alias("sym"), (pl.col("fy_end") + pl.duration(days=90)).alias("date"),
        pl.col("net_income").alias("ni"), "shares").sort("date")
    ms, a = tm(lambda: ind.join_asof(f, on="date", by="sym", strategy="backward"), reps)
    emit(eng, "asof_join_in", ms, a["ni"].count(), threads)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["duckdb", "polars"])
    ap.add_argument("--root", default=str(Path.home() / ".cache/market-warehouse"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.engine == "polars":
        os.environ["POLARS_MAX_THREADS"] = str(a.threads)   # must precede the import
        run_polars(a.root, a.reps, a.threads)
    else:
        run_duckdb(a.root, a.reps, a.threads)
