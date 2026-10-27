# global-market-kdb

A kdb+/q implementation of my cross-sectional equity signal study. It covers:

- the data layer: a date-partitioned HDB of daily bars,
- twelve signals at four horizons,
- the statistics: Newey-West t, BH-FDR, Fama-MacBeth, cost-adjusted quintile spreads, the Deflated Sharpe ratio and a purged walk-forward combination.

A parity test checks every output table against the original Python, row by row.

The Python originals live in
[`herrrickshaw/global-market-scanners`](https://github.com/herrrickshaw/global-market-scanners)
(`signal_research.py`, `signal_validation.py`; method in `SIGNAL_RESEARCH.md`).
They are vendored verbatim in [`python/upstream/`](python/upstream/) as the
oracle. The warehouse they read is described in
[`herrrickshaw/global-market-data`](https://github.com/herrrickshaw/global-market-data).

> Research code, not investment advice. The committed sample is synthetic, and
> no number printed from it says anything about any market.

## Parity status

The q study (`q/`) is checked against upstream's own `study_market()`, run unchanged by `python/study_py.py` on the same input. The pass rule is |q − py| ≤ 1e-10 + 1e-8·|py|, with identical nulls, labels and counts.

| Input | ic_series | ic_table | fama_macbeth | costs + DSR | walk_forward | health |
|---|---|---|---|---|---|---|
| Synthetic sample: 120 names, 2018–2022, 1 market | 10,032 rows, max diff 0 | 48 rows, 1.4e-13 | 11 rows, 4.3e-14 | 33 rows, 2.3e-13 | 5 folds, 1.4e-17 | 11 rows, 4.4e-16 |
| Real slice: US + India, 200 names each, 2016–2022 | 27,552 rows, max diff 0 | 96 rows, 1.4e-12 | 22 rows, 1.9e-13 | 66 rows, 2.3e-13 | 10 folds, 2.8e-17 | 22 rows, 8.9e-16 |
| Synthetic sample through PyKX (in-process) | identical to the first row | | | | | |

All rows **PASS**. They were run on KDB-X 5.0 Community (2026.10.23) with PyKX 4.2.0 on macOS arm64. `pytest -q` reports 6 passed.

**One documented exemption.** `lowvol` is −(63-day volatility), and the same 63-day volatility is one of the two neutralisation exposures. So rank(lowvol) is a linear function of rank(vol), and the neutralised residual is floating-point round-off: the median |residual| is 2e-16, against 0.24 for `rev5`. Upstream's `neutral_ic`/`neutral_t` for `lowvol` are therefore the IC of noise, and no two correct implementations will agree on them. Parity compares only the null pattern for those two cells. This is a finding about the upstream study, not a gap in the port.

## What is computed

For each market, daily bars are put on the full date × symbol grid. Close is forward-filled for at most 3 days. The panel is then sampled on the last trading day of each week (pandas `W-FRI`), from `-start`.

| Signal | Definition (corporate-action guarded) |
|---|---|
| `rev5` | −(5-day return) |
| `rsi_os` | −RSI-14 (Wilder EWM, α = 1/14, on simple returns) |
| `mom21`, `mom126` | 21- and 126-day return |
| `mom12_1` | 12-1 momentum: C[t−21]/C[t−252] − 1 |
| `trend` | close / EMA50 − 1 |
| `golden` | EMA50 / EMA200 − 1 |
| `hi52` | close / 252-day max high (≥ 200 obs) |
| `lowvol` | −(63-day std of returns) |
| `illiq` | Amihud: 63-day mean of \|r\| / dollar volume |
| `volsurge` | log(20-day mean volume / 126-day mean volume) |
| `zone` | the mailer's zone state: +1 if C > EMA20 > EMA50, −1 if C < EMA50, else 0 |

**Forward returns** are measured at 5, 10, 21 and 63 days, entered at the next close.

**Liquidity gate (strictly trailing):** a name must rank in the top `n` by 63-day median dollar volume and have at least 240 observed days in the last 252.

Studies, per market (`q/study.q`):

| Output | Content |
|---|---|
| `ic_table` | mean rank IC, Newey-West t (⌈h/5⌉ lags) and p, annualised ICIR, hit rate; IC after neutralising on liquidity + volatility ranks; bull/bear regime IC, where bear means fewer than 45% of gated names are above their 200-day mean. Across markets: BH-FDR q-values and the Harvey-Liu-Zhu \|t\| ≥ 3 flag |
| `health` | trailing-52-week 10d IC vs full history → HEALTHY / DECAYING / FLIPPED / NOISE |
| `fama_macbeth_10d` | per-date joint regression of z-ranked forward return on all z-ranked signals; mean slope and Newey-West t |
| `costs` | quintile long-short spread, long-only excess, turnover, break-even cost, net-of-cost annualised return and Sharpe, on a non-overlapping grid at 5/21/63d. Deflated Sharpe probability across every strategy tried |
| `walk_forward_10d` | expanding window, purged with a 3-date embargo; out-of-sample IC of best-single vs IC-weighted vs ridge combinations chosen on training data only |
| `ic_series` | per-date rank IC for every signal × horizon |

The statistics are implemented in q from scratch (`q/stats.q`):

- the Student-t p-value, through a regularised incomplete beta (continued fraction);
- the normal CDF, through a regularised incomplete gamma;
- the normal quantile, using Acklam's approximation plus two Halley steps;
- a Lanczos log-gamma;
- BH-FDR, the Deflated Sharpe ratio and OLS.

`tests/test_parity.py` checks the special functions against scipy to a relative 1e-12.

## Layout

```
q/schema.q     daily bar table (sym, mkt, OHLCV), partitioned by date
q/load.q       interchange CSV -> date-partitioned HDB (.Q.dpft, `p#sym per partition)
q/signals.q    .sig.build: the gated signal panel (12 signals, 4 horizons, exposures)
q/stats.q      .st: Newey-West, BH-FDR, t / normal distributions, Deflated Sharpe, OLS
q/study.q      .study.run: the full study, every market, cross-market post-processing
q/run.q        HDB -> one CSV per output table
python/upstream/          vendored upstream oracle (verbatim, commit cb70eca)
python/study_py.py        runs the oracle on the interchange CSV
python/parity.py          q vs oracle, all six tables
python/pykx_parity.py     the same check with q embedded in Python through PyKX
python/make_sample.py     generates data/sample/ohlcv.csv (seeded, synthetic)
python/extract_sample.py  real slice from the Parquet warehouse -> data/local/ (gitignored)
tests/test_parity.py      pytest: EWM recurrence, special functions, full parity
```

## Porting notes

The q code follows pandas semantics exactly where they are easy to get wrong:

- `ewm(adjust=False)` decays the old weight across null inputs.
- `rolling(n, min_periods=m)` ignores nulls.
- `rank` averages ties.
- In q, `0|0n` is `0` and `0n<x` is true, so `clip(lower=0)` and every null comparison is guarded explicitly.

q and KDB-X details that bit:

- A line containing only `/` opens a block comment that runs until a line containing only `\`.
- Names inside q-sql column expressions resolve in the root namespace at run time, so `.sig.build` calls its helpers fully qualified.
- `all` is a reserved word, so it can't be a function name; KDB-X also reserves the `-n` command-line flag.
- In KDB-X, `system "cd"` returns a string rather than a list of strings.

## Running it

KDB-X Community Edition is free for personal and commercial use within KX's [usage limits](https://code.kx.com/licensing/usage-restrictions.html) (16 GB RAM, a single instance, 24 cores). Register at the KX Developer Center and run the install command it gives you ([install guide](https://code.kx.com/get_started/kdb-x-install.html)). The installer puts `q` in `~/.kx/bin` and adds it to your PATH. Otherwise, set `Q=~/.kx/bin/q`.

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python python/parity.py              # synthetic sample, q vs upstream
.venv/bin/python -m pytest -q
```

q directly:

```bash
q q/load.q -csv data/sample/ohlcv.csv -hdb db
q q/run.q -hdb db -univ 100 -start 2019.01.01 -out out/q
```

Through PyKX. It needs the same licence, because unlicensed PyKX cannot execute q. Point it at the licence with `QLIC`, or it stops at an interactive prompt:

```bash
QLIC=~/.kx .venv/bin/python python/pykx_parity.py
```

On real data, if you have the warehouse locally:

```bash
.venv/bin/python python/extract_sample.py --mkt US --names 200 --from 2016-06-01 --to 2022-12-31 --out data/local/US.csv
.venv/bin/python python/parity.py --csv data/local/US.csv --n 150 --start 2017-07-01 --work out/us
```

## Data

`data/sample/ohlcv.csv` holds 120 synthetic names on business days from 2018 to 2022, 142,483 rows. It is seeded and regenerated by `python/make_sample.py`. It deliberately includes missing rows, late listings and delistings, zero-volume days, unadjusted ±jumps and `high == close` ties, so parity exercises the null, guard and tie paths.

Real warehouse prices come from market-data vendors and are not redistributed here; `extract_sample.py` writes them to the gitignored `data/local/`. Two deviations from upstream's loader:

- panels are float64 rather than float32;
- the yearly top-N pre-filter is replaced by picking the `--names` most liquid symbols for the extracted window.

The per-date liquidity gate inside the study is identical to upstream.

## Licence

Code: MIT (see `LICENSE`). The vendored files in `python/upstream/` are from my own upstream repository. kdb+/KDB-X and PyKX are KX products under KX's own licences, and nothing from KX is vendored here.
