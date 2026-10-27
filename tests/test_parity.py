"""pytest -q   (q-dependent tests skip when q / a licensed PyKX is not available)"""
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
import reference as ref  # noqa: E402

SAMPLE = ROOT / "data/sample/ohlcv.csv"
SCANNERS = Path.home() / "repos/global-market-scanners"


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


@pytest.mark.parametrize("a,minp", [(1 / 14, 14), (2 / 51, 0), (2 / 201, 0)])
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


@pytest.mark.skipif(not (SCANNERS / "signal_research.py").exists(), reason="no global-market-scanners checkout")
def test_reference_matches_upstream():
    r = subprocess.run([sys.executable, str(ROOT / "python/check_upstream.py"), str(SAMPLE),
                        "--scanners", str(SCANNERS)], capture_output=True, text=True, check=True)
    worst = float(r.stdout.strip().splitlines()[-1].split("=")[-1])
    assert worst < 1e-3      # float32 rounding in upstream's illiq only; see README


@pytest.mark.skipif(parity.find_q() is None, reason="q not installed")
def test_q_matches_reference(tmp_path):
    ic_q = parity.run_q(parity.find_q(), SAMPLE, "SYN", 100, "2019-01-01", tmp_path)
    ok, table = parity.compare(ic_q, ref.run(SAMPLE, 100, "2019-01-01"), 1e-10)
    assert ok, table.to_string()
