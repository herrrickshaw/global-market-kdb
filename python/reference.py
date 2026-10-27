#!/usr/bin/env python3
"""Python reference for the q port: five signals, 21-day forward return,
trailing liquidity gate, and per-formation-date cross-sectional rank IC.

Vendored from herrrickshaw/global-market-scanners @ cb70eca
(signal_research.py::build_signals, signal_validation.py::rank_ic_series and
::newey_west_t), cut down to the pieces this repo ports. Formulas, windows and
min-periods are unchanged. One deliberate difference: upstream casts the
panels to float32 to save memory on the full warehouse; this reference keeps
float64, so the q port (also float64) can be compared exactly.
python/check_upstream.py measures the effect of that cast on the sample.

Run:
  python python/reference.py data/sample/ohlcv.csv --n 100 --start 2019-01-01 \
      --out out/ic_py.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

SIGNALS = ("rsi_os", "hi52", "rev5", "golden", "illiq")
H = 21                # forward horizon in trading days
MIN_NAMES = 30        # dates with fewer valid names give NaN IC


def load_panels(csv: str | Path) -> dict[str, pd.DataFrame]:
    """Date x symbol Close/High/Volume panels (upstream load_market, float64)."""
    df = pd.read_csv(csv, parse_dates=["date"])
    df = df[df["close"] > 0]
    out = {}
    for col, name in (("close", "Close"), ("high", "High"), ("volume", "Volume")):
        out[name] = (df.pivot_table(index="date", columns="sym", values=col, aggfunc="last")
                       .sort_index().astype("float64"))
    return out


def build(p: dict[str, pd.DataFrame], n: int, start: str):
    """Signals and the 21d forward return, sampled weekly and gated (upstream build_signals)."""
    C = p["Close"].ffill(limit=3)
    Hi = p["High"].reindex_like(C)
    V = p["Volume"].reindex_like(C).fillna(0)
    r1 = C.pct_change(fill_method=None)
    bad = ((r1 < -0.5) | (r1 > 1.0)).astype("float64")     # corporate-action guard
    dv = C * V

    def clean(x: pd.DataFrame, lookback: int) -> pd.DataFrame:
        hit = bad.rolling(lookback, min_periods=1).max() > 0
        return x.mask(hit)

    up = r1.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-r1).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi14 = 100 - 100 / (1 + up / dn.where(dn > 0))       # Wilder RSI on returns
    ema50 = C.ewm(span=50, adjust=False).mean()
    ema200 = C.ewm(span=200, adjust=False).mean()
    sig = {
        "rsi_os": clean(-rsi14, 30),
        "hi52": clean(C / Hi.rolling(252, min_periods=200).max(), 253),
        "rev5": clean(-(C / C.shift(5) - 1), 6),
        "golden": clean(ema50 / ema200 - 1, 200),
        "illiq": clean((r1.abs() / dv.replace(0, np.nan)).rolling(63, min_periods=50).mean(), 64),
    }

    f = C.shift(-(H + 1)) / C.shift(-1) - 1                 # enter next close
    fbad = bad[::-1].rolling(H + 1, min_periods=1).max()[::-1].shift(-1) > 0
    fwd = f.mask(fbad)

    mdv63 = dv.rolling(63, min_periods=40).median()
    fridays = C.resample("W-FRI").last().index
    idx = C.index[C.index.searchsorted(fridays, side="right") - 1]
    idx = pd.DatetimeIndex(sorted(set(idx[idx >= pd.Timestamp(start)])))
    hist = C.notna().rolling(252, min_periods=1).sum()
    gate = (mdv63.loc[idx].rank(axis=1, ascending=False) <= n) & (hist.loc[idx] >= 240)

    S = {k: v.loc[idx].where(gate) for k, v in sig.items()}
    return S, fwd.loc[idx].where(gate)


def rank_ic_series(signal: pd.DataFrame, fwd: pd.DataFrame,
                   min_names: int = MIN_NAMES) -> tuple[pd.Series, pd.Series]:
    """Per-date Spearman IC over names where both values exist (upstream, plus the count)."""
    s, f = signal.align(fwd, join="inner")
    both = s.notna() & f.notna()
    rs = s.where(both).rank(axis=1)
    rf = f.where(both).rank(axis=1)
    rs = rs.sub(rs.mean(axis=1), axis=0)
    rf = rf.sub(rf.mean(axis=1), axis=0)
    num = (rs * rf).sum(axis=1)
    den = np.sqrt((rs ** 2).sum(axis=1) * (rf ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    cnt = both.sum(axis=1)
    ic[cnt < min_names] = np.nan
    return ic, cnt


def newey_west_t(x, lags: int) -> tuple[float, float]:
    """Mean and HAC (Newey-West, Bartlett) t-stat (upstream, without the p-value)."""
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(v)
    if n < 3:
        return float("nan"), float("nan")
    mu = v.mean()
    e = v - mu
    var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (e[k:] @ e[:-k]) / n
    return float(mu), float(mu / math.sqrt(max(var, 1e-300) / n))


def run(csv, n: int, start: str) -> pd.DataFrame:
    S, F = build(load_panels(csv), n, start)
    rows = []
    for k in SIGNALS:
        ic, cnt = rank_ic_series(S[k], F)
        rows.append(pd.DataFrame({"date": ic.index, "signal": k, "ic": ic.values,
                                  "n": cnt.values}))
    return pd.concat(rows, ignore_index=True)


def summary(ic: pd.DataFrame, lags: int = math.ceil(H / 5)) -> pd.DataFrame:
    out = []
    for k, g in ic.groupby("signal", sort=False):
        mu, t = newey_west_t(g["ic"], lags)
        out.append({"signal": k, "mean_ic": mu, "nw_t": t, "n_dates": int(g["ic"].notna().sum())})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=100, help="universe size for the liquidity gate")
    ap.add_argument("--start", default="2019-01-01", help="first formation date")
    ap.add_argument("--out", default="out/ic_py.csv")
    a = ap.parse_args()
    ic = run(a.csv, a.n, a.start)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ic.to_csv(a.out, index=False, date_format="%Y-%m-%d", float_format="%.17g")
    print(summary(ic).to_string(index=False))


if __name__ == "__main__":
    main()
