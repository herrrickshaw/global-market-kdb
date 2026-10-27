"""pytest -q   (q-dependent tests skip when q is not installed)"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import parity  # noqa: E402
import study_py  # noqa: E402

SAMPLE = ROOT / "data/sample/ohlcv.csv"
needs_q = pytest.mark.skipif(parity.find_q() is None, reason="q not installed")


def ewm_mirror(x, a, minp):
    """Python mirror of .sig.ewstep / .sig.ewm in q/signals.q."""
    w, ow, nobs, out = np.nan, 1.0, 0, []
    for c in x:
        nobs += not np.isnan(c)
        if np.isnan(w):
            if not np.isnan(c):
                w, ow = c, 1.0
        else:
            ow *= 1 - a
            if not np.isnan(c):
                w, ow = (w if w == c else (ow * w + a * c) / (ow + a)), 1.0
        out.append(w if nobs >= max(minp, 1) else np.nan)
    return np.array(out)


@pytest.mark.parametrize("a,minp", [(1 / 14, 14), (2 / 21, 0), (2 / 51, 0), (2 / 201, 0)])
def test_ewm_recurrence_matches_pandas(a, minp):
    """The state recurrence the q port uses reproduces pandas ewm(adjust=False) with nulls."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    x[rng.random(400) < 0.2] = np.nan
    x[:7] = np.nan
    p = pd.Series(x).ewm(alpha=a, adjust=False, min_periods=minp).mean().to_numpy()
    m = ewm_mirror(x, a, minp)
    assert (np.isnan(p) == np.isnan(m)).all()
    assert np.nanmax(np.abs(p - m)) < 1e-14


@needs_q
def test_q_special_functions_match_scipy(tmp_path):
    from scipy import special, stats
    script = tmp_path / "st.q"
    script.write_text(
        f'system "l {ROOT}/q/stats.q"\nsystem "P 17"\n'
        '-1 " " sv string .st.tp\'[0.5 1.96 3.2 8.0 0.01;200 50 10 99 5];\n'
        '-1 " " sv string .st.ncdf each -9.6 -3 -0.5 0 0.7 2.5 6;\n'
        '-1 " " sv string .st.nppf each 0.9696969696969697 0.98885 0.5 0.001 0.3;\n'
        '-1 " " sv string .st.lgamma each 0.5 2.5 10 100.5;\nexit 0\n')
    out = subprocess.run([parity.find_q(), str(script), "-q"], capture_output=True, text=True,
                         check=True).stdout.split("\n")
    got = [np.array(line.split(), dtype=float) for line in out[:4]]
    want = [2 * stats.t.sf([0.5, 1.96, 3.2, 8.0, 0.01], [200, 50, 10, 99, 5]),
            stats.norm.cdf([-9.6, -3, -0.5, 0, 0.7, 2.5, 6]),
            stats.norm.ppf([0.9696969696969697, 0.98885, 0.5, 0.001, 0.3]),
            special.gammaln([0.5, 2.5, 10, 100.5])]
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-12, atol=1e-15)


@needs_q
def test_q_study_matches_upstream(tmp_path):
    res_q = parity.run_q(parity.find_q(), SAMPLE, 100, "2019-01-01", tmp_path)
    res_py = study_py.run(SAMPLE, 100, "2019-01-01")
    rows = [parity.compare(k, res_q.get(k), res_py.get(k), 1e-10, 1e-8) for k in parity.KEYS]
    table = pd.DataFrame(rows)
    assert table["pass"].all(), table.to_string()
    assert (table["rows"] > 0).all()
