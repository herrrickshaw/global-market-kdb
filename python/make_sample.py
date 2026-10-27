#!/usr/bin/env python3
"""Generate the committed synthetic sample: data/sample/ohlcv.csv.

Seeded and deterministic. The prices are synthetic (factor + idiosyncratic
random walk); they carry no information about any real market and exist only so
the q port and the Python reference can be compared on identical input.

The generator deliberately includes the awkward cases the signal code has to
handle, so parity covers them:
  * missing rows (exercises the 3-day forward-fill limit and null propagation)
  * late listings and early delistings (exercises min-history and min-periods)
  * zero-volume days (dollar volume 0 -> excluded from Amihud, kept in median)
  * a few unadjusted -60% / +150% jumps (exercises the corporate-action guard)
  * High == Close on some days (ties in the 52-week-high signal)

Run:  python python/make_sample.py            # writes data/sample/ohlcv.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample" / "ohlcv.csv"
SEED = 20261027
N_NAMES = 120
DATES = pd.bdate_range("2018-01-01", "2022-12-30")


def main() -> None:
    rng = np.random.default_rng(SEED)
    T, N = len(DATES), N_NAMES
    syms = [f"S{i:03d}" for i in range(1, N + 1)]

    mkt = rng.normal(0.0003, 0.010, T)
    beta = rng.uniform(0.5, 1.5, N)
    ivol = rng.lognormal(np.log(0.018), 0.4, N)
    eps = rng.normal(0, 1, (T, N)) * ivol
    # mild short-horizon reversal in the idiosyncratic part (AR(1), phi < 0)
    for t in range(1, T):
        eps[t] += -0.05 * eps[t - 1]
    r = mkt[:, None] * beta + eps

    # unadjusted corporate-action jumps on a handful of names
    for j in rng.choice(N, 6, replace=False):
        t = rng.integers(300, T - 50)
        r[t, j] = np.log(0.4) if rng.random() < 0.5 else np.log(2.5)

    close = 20 * np.exp(rng.normal(0, 0.8, N)) * np.exp(np.cumsum(r, axis=0))
    spread = np.abs(rng.normal(0, 0.008, (T, N)))
    high = close * (1 + spread)
    tie = rng.random((T, N)) < 0.10
    high = np.where(tie, close, high)
    low = close * (1 - np.abs(rng.normal(0, 0.008, (T, N))))
    open_ = np.clip(close * (1 + rng.normal(0, 0.004, (T, N))), low, high)
    vscale = rng.lognormal(np.log(2e5), 1.0, N)
    volume = np.round(vscale * rng.lognormal(0, 0.5, (T, N)))
    volume[rng.random((T, N)) < 0.01] = 0

    df = pd.DataFrame({
        "date": np.repeat(DATES.values, N),
        "mkt": "SYN",
        "sym": np.tile(syms, T),
        "open": open_.ravel(), "high": high.ravel(), "low": low.ravel(),
        "close": close.ravel(), "volume": volume.ravel(),
    })

    keep = rng.random(len(df)) > 0.01                       # ~1% missing rows
    life = pd.DataFrame({"sym": syms,
                         "first": DATES[np.where(rng.random(N) < 0.15, rng.integers(0, T // 2, N), 0)],
                         "last": DATES[np.where(rng.random(N) < 0.10, rng.integers(T // 2, T, N), T - 1)]})
    df = df.merge(life, on="sym")
    keep &= (df["date"] >= df["first"]) & (df["date"] <= df["last"])
    df = df.loc[keep, ["date", "mkt", "sym", "open", "high", "low", "close", "volume"]]

    for c in ("open", "high", "low", "close"):
        df[c] = df[c].round(4)
    df["volume"] = df["volume"].astype("int64")
    df = df.sort_values(["date", "sym"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df):,} rows, {df['sym'].nunique()} names, "
          f"{df['date'].min():%Y-%m-%d}..{df['date'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
