# Vendored upstream oracle

`signal_research.py` and `signal_validation.py` are byte-for-byte copies of
[`herrrickshaw/global-market-scanners`](https://github.com/herrrickshaw/global-market-scanners)
at commit `cb70eca`, taken with `git show cb70eca:<file>`. `apply_costs.py` is a
stub holding only the `COSTS_PCT` table that `signal_research.py` imports.

They are the parity oracle. `python/study_py.py` runs upstream's own
`study_market()` and main()-level post-processing on the interchange CSV, and the
q port in `q/` must reproduce its output. Do not edit these files: re-vendor
them from a newer upstream commit instead.
