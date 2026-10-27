# global-market-kdb

A kdb+/q port of the core of my equity signal research stack: a date-partitioned
daily OHLCV database, five cross-sectional signals, and per-formation-date
Spearman rank IC against the 21-day forward return. A parity test checks the
q output row by row against the original Python.

The Python originals live in
[`herrrickshaw/global-market-scanners`](https://github.com/herrrickshaw/global-market-scanners)
(`signal_research.py`, `signal_validation.py`; method in `SIGNAL_RESEARCH.md`).
The warehouse they read is described in
[`herrrickshaw/global-market-data`](https://github.com/herrrickshaw/global-market-data).

> Research code, not investment advice. The committed sample is synthetic, and
> no IC printed from it says anything about any market.

## Status

| Check | Result |
|---|---|
| `python/reference.py` (vendored, float64) vs upstream `signal_research.py`, committed synthetic sample | identical per-date IC for 4 of 5 signals; `illiq` differs by at most 2.9e-4 (upstream casts panels to float32) |
| Same check on a real 150-name US slice, 2018–2020, from the warehouse | identical per-date IC for all 5 signals (the warehouse already stores float32) |
| **q (`q/signals.q`) vs `python/reference.py`**, synthetic sample, via HDB + `q/run.q` | **PASS**: 105 formation dates × 5 signals, same valid-name counts and null pattern, max per-date \|ΔIC\| = 0 (tolerance 1e-10) |
| Same, real 150-name US slice, 2018–2020 | **PASS**, max per-date \|ΔIC\| = 0 |
| Same, synthetic sample, in-process via PyKX (`python/pykx_parity.py`) | **PASS**, max per-date \|ΔIC\| = 0 |

Run on KDB-X 5.0 Community (2026.10.23), PyKX 4.2.0, macOS arm64. `pytest -q`: 5 passed.

## Layout

```
q/schema.q          daily bar table: sym, mkt, open, high, low, close, volume; partitioned by date
q/load.q            interchange CSV -> date-partitioned HDB (.Q.dpft, `p#sym per partition)
q/signals.q         .sig.build (signals, forward return, liquidity gate), .sig.ic, .sig.summary
q/run.q             HDB -> per-date IC CSV + Newey-West summary
python/reference.py       Python reference (vendored subset of upstream, float64)
python/parity.py          q vs reference, every (date, signal) row
python/pykx_parity.py     the same check through PyKX (q embedded in Python)
python/check_upstream.py  reference vs the original upstream code
python/make_sample.py     generates data/sample/ohlcv.csv (seeded, synthetic)
python/extract_sample.py  real slice from the Parquet warehouse -> data/local/ (gitignored)
tests/test_parity.py      pytest wrapper for the above
```

## What is computed

Daily bars are put on the full date × symbol grid. Close is forward-filled for
at most 3 days. The signals are then sampled on the last trading day of each
week (pandas `W-FRI`), starting at `-start`.

| Signal | Definition | q |
|---|---|---|
| `rsi_os` | −RSI-14 (Wilder EWM, α = 1/14, on simple returns): oversold scores high | `.sig.rsi`, `.sig.ewm` |
| `hi52` | close / 252-day max of high (≥ 200 obs) | `.sig.rmax` |
| `rev5` | −(5-day return) | `xprev` |
| `golden` | EMA50 / EMA200 − 1 | `.sig.ewm` |
| `illiq` | 63-day mean of \|r\| / dollar volume, Amihud (≥ 50 obs, zero-volume days excluded) | `.sig.rmean` |

- **Corporate-action guard:** a daily move outside [−50%, +100%] nulls every signal whose lookback window contains it. It also nulls any forward window that contains it.
- **Forward return:** 21 days, entered at the next close: `C[t+22]/C[t+1] − 1`.
- **Liquidity gate (strictly trailing):** keep the top `n` names by 63-day median dollar volume, with at least 240 of the last 252 days observed.
- **IC:** Spearman (average ranks) over names where both the signal and the forward return exist. A date with fewer than 30 such names gives a null IC.
- **Summary:** mean IC with a Newey-West t-stat, Bartlett kernel, ⌈21/5⌉ = 5 lags.

The q code follows pandas semantics exactly where they are easy to get
wrong. `ewm(adjust=False)` decays the old weight across null inputs.
`rolling(n, min_periods=m)` ignores nulls. `rank` averages ties. In q,
`0|0n` is `0` and `0n<x` is true, so `clip(lower=0)` and the null guards
are explicit. `tests/test_parity.py` pins the EWM recurrence against pandas.

q details that bit during the port:

- A line containing only `/` opens a block comment that runs until a line containing only `\`.
- Names inside q-sql column expressions are looked up in the root namespace at run time, not in the namespace the function was defined in. So `.sig.build` qualifies its helpers (`.sig.ewm`, `.sig.clean`, …).
- KDB-X reserves the `-n` command-line flag, so the universe size is passed as `-univ`.

## Running it

KDB-X Community Edition is free for personal and commercial use within KX's
[usage limits](https://code.kx.com/licensing/usage-restrictions.html)
(16 GB RAM, a single instance, 24 cores). Register at the KX Developer
Center. The install command it gives you carries your licence key; see the
[install guide](https://code.kx.com/get_started/kdb-x-install.html).

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python python/parity.py                 # synthetic sample, q vs Python
.venv/bin/python -m pytest -q
```

q directly:

```bash
q q/load.q -csv data/sample/ohlcv.csv -hdb db
q q/run.q -hdb db -mkt SYN -univ 100 -start 2019.01.01 -out out/ic_q.csv
```

The KDB-X installer puts `q` in `~/.kx/bin` and adds it to your shell's PATH. Otherwise, set `Q=~/.kx/bin/q`.

Through PyKX (`pip install pykx`). It needs the same licence, because unlicensed PyKX cannot execute q. Point it at the licence with `QLIC`, or it stops at an interactive licence prompt:

```bash
QLIC=~/.kx .venv/bin/python python/pykx_parity.py
```

On real data, if you have the warehouse locally:

```bash
.venv/bin/python python/extract_sample.py --mkt US --names 150 --from 2018-01-01 --to 2020-12-31
.venv/bin/python python/parity.py --csv data/local/US.csv --n 100 --work out/us
```

## Data

`data/sample/ohlcv.csv` holds 120 synthetic names on business days from 2018 to 2020,
87,441 rows. It is seeded and regenerated by `python/make_sample.py`. It
deliberately includes missing rows, late listings and delistings,
zero-volume days, unadjusted ±jumps and `high == close` ties, so the
parity test exercises the null, guard and tie paths. Real warehouse prices
come from market-data vendors and are not redistributed here;
`extract_sample.py` writes them to the gitignored `data/local/`.

## Scope and next steps

This is the first slice of the upstream study: signals and rank IC. Not yet
ported: the 5/10/63-day horizons, liquidity/volatility neutralisation, the
regime split, Fama-MacBeth, cost-adjusted quantile spreads, the Deflated
Sharpe ratio and the purged walk-forward combination.

## Licence

Code: MIT (see `LICENSE`). kdb+/KDB-X and PyKX are KX products under KX's own
licences. Nothing from KX is vendored here.
