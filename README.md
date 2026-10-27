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
| Same US + India slice, read from the warehouse Parquet by q (`pq`) and by DuckDB | 27,552 rows, max diff 0 | 96 rows, 1.4e-12 | 22 rows, 1.9e-13 | 66 rows, 2.8e-13 | 10 folds, 2.8e-17 | 22 rows, 8.9e-16 |
| Synthetic sample through PyKX (in-process) | identical to the first row | | | | | |

All rows **PASS**. They were run on KDB-X 5.0 Community (2026.10.23) with PyKX 4.2.0 on macOS arm64. `pytest -q` reports 10 passed.

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

## Beyond upstream

These additions have no upstream counterpart, so they're checked against independent computations instead of an oracle (`tests/test_extras.py`).

| Addition | q | Check |
|---|---|---|
| Read the Parquet warehouse in q, with no Python or CSV step | `q/pqload.q` (KDB-X `pq` module) | the full study on a `pq`-loaded HDB matches upstream on DuckDB-read data (parity row above) |
| Point-in-time fundamentals: annual net income and shares, usable 90 days after FY end, joined with `aj` → earnings yield and its IC | `.ext.fund`, `.ext.ep`, `.ext.pitic` | `aj` equals pandas `merge_asof(direction="backward")`; every joined record has availability date ≤ formation date |
| Corporate actions: back-adjust bars for splits/bonuses, then rerun the study on adjusted vs raw bars | `.ext.cafile`, `.ext.adjust`, `.ext.caeffect` | a toy split adjusts to a flat series; on the India slice, 103 events cut price-jump guard hits from 41 to 5 |
| Spread-based costs: Corwin-Schultz (2012) high-low spread per name and date; each rebalance pays half the spread on every name traded | `.sig.csd`, `.ext.tca` | the estimator equals a NumPy implementation to 1e-12 |
| Overnight vs intraday: each forward return split into close→open and open→close log legs, IC of every signal on each leg | `.sig.legs`, `.ext.legsic` | the legs sum to the forward log return (max deviation 4e-15 on real data) |

What the extras showed on the real slice (US + India, 200 names each, formation dates 2017–2022). This is exploratory, from one sample of one universe, with no multiple-testing correction, and not investment advice:

- **Reversal is an overnight effect** in both markets. The 5-day reversal IC's Newey-West t is 3.97 on the overnight leg vs 0.30 intraday in India, and 3.81 vs −1.00 in the US. For context, Lou, Polk & Skouras (2019, *JFE*, "A tug of war: overnight versus intraday expected returns") document that different cross-sectional strategies earn their returns in different parts of the day.
- **India low-volatility flips by leg:** t = −9.1 overnight vs +4.8 intraday.
- **Spread costs:** the Corwin-Schultz median is 73 bps for the India slice and 49 bps for the US slice. The estimator is known to overstate spreads for liquid large caps, so its net returns are a pessimistic bound, not a forecast.
- **Earnings yield** (India, 90-day lag) has no significant IC at any horizon on this slice: \|t\| < 0.4, with 55% coverage.

## Benchmark

`bench/` times five of the study's own operations on the full US warehouse (16.7M rows) in q, DuckDB and Polars, and checks that the engines agree on every answer. Results and caveats: [`bench/RESULTS.md`](bench/RESULTS.md).

Minimum of 5 runs on an 8 GB M1. q runs single-threaded unless the work is split with `peach`; DuckDB and Polars are shown on 4 threads:

| operation | q | DuckDB | Polars |
|---|---|---|---|
| load US Parquet (16.7M rows) | 1,171 ms (`peach`) | 324 ms | 288 ms |
| median dollar volume by symbol, year | 746 ms (`peach`) | 237 ms | 975 ms |
| 5d return + 63-row mean \|r\| by symbol | 841 ms | 3,022 ms | 552 ms |
| rank by date | 1,857 ms (date-sorted) | 5,224 ms | 714 ms |
| as-of join, India bars to fundamentals | 67 ms | 538 ms | 70 ms |

q leads on the as-of join, tied with Polars. It's well ahead of DuckDB on the time-series and cross-sectional steps, and level with single-threaded Polars on the windows (837 ms). DuckDB is fastest at the grouped median, and DuckDB and Polars both load Parquet 3–4× faster than `pq`. The machine was swapping, so treat differences under about 2× as noise.

## Layout

```
q/schema.q     daily bar table (sym, mkt, OHLCV), partitioned by date
q/load.q       interchange CSV -> date-partitioned HDB (.Q.dpft, `p#sym per partition)
q/signals.q    .sig.build: the gated signal panel (12 signals, 4 horizons, exposures)
q/stats.q      .st: Newey-West, BH-FDR, t / normal distributions, Deflated Sharpe, OLS
q/study.q      .study.run: the full study, every market, cross-market post-processing
q/run.q        HDB -> one CSV per output table
q/pqload.q     Parquet warehouse -> HDB, natively with the KDB-X pq module
q/extras.q     .ext: spread costs, point-in-time fundamentals, corporate actions, overnight/intraday
q/ext_run.q    HDB (+ fundamentals, corporate actions) -> the extras' tables
bench/         q vs DuckDB vs Polars (bench.q, bench_py.py, run.py, RESULTS.md)
python/upstream/          vendored upstream oracle (verbatim, commit cb70eca)
python/study_py.py        runs the oracle on the interchange CSV
python/parity.py          q vs oracle, all six tables
python/pykx_parity.py     the same check with q embedded in Python through PyKX
python/make_sample.py     generates data/sample/ohlcv.csv (seeded, synthetic)
python/extract_sample.py  real slice from the Parquet warehouse -> data/local/ (gitignored)
tests/test_parity.py      pytest: EWM recurrence, special functions, full parity
tests/test_extras.py      pytest: spread estimator, leg identity, aj vs merge_asof, split adjustment
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
- Built-in names can't be reused for globals or locals: `all`, `read1`, `cov`, `in`, `from` and `to` all failed with `'assign`. KDB-X also reserves the `-n` command-line flag.
- A completeness test on a trailing window has to be shifted together with the window sum, not after it. The leg-identity test caught this one.
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

Straight from the Parquet warehouse, with the extras:

```bash
q q/pqload.q -root ~/.cache/market-warehouse/ohlcv -mkts US IN -from 2016.06.01 -to 2022.12.31 -names 200 -hdb db
q q/run.q -hdb db -univ 150 -start 2017.07.01 -out out/q
q q/ext_run.q -hdb db -univ 150 -start 2017.07.01 -out out/ext -fund ~/.cache/market-warehouse/fundamentals_history/IN_screener_only_backup.parquet -ca ~/.cache/market-warehouse/adjustment_factors.parquet
.venv/bin/python python/parity.py --parquet ~/.cache/market-warehouse/ohlcv --mkts US IN --n 150 --start 2017-07-01
.venv/bin/python bench/run.py
```

From a CSV extract instead:

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
