"""Checks for q/extras.q and the spread / leg helpers in q/signals.q.
These are new features with no upstream counterpart, so each is checked
against an independent pandas/numpy computation or an identity."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import parity  # noqa: E402

pytestmark = pytest.mark.skipif(parity.find_q() is None, reason="q not installed")
LOAD = "".join(f'system "l {ROOT}/q/{f}.q";' for f in ("stats", "signals", "study", "extras"))


def q_json(tmp_path, body: str):
    """Run q code that leaves its result in `r`, return it via JSON."""
    s = tmp_path / "t.q"
    s.write_text(LOAD + '\nsystem "P 0";\n' + body + '\n-1 .j.j r;\nexit 0\n')
    p = subprocess.run([parity.find_q(), str(s), "-q"], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr + p.stdout
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_corwin_schultz_matches_numpy(tmp_path):
    rng = np.random.default_rng(3)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    h, l = c * (1 + rng.uniform(0, 0.02, 60)), c * (1 - rng.uniform(0, 0.02, 60))
    h[10], l[20] = np.nan, np.nan
    fmt = lambda a: " ".join("0n" if np.isnan(v) else repr(float(v)) for v in a)
    got = q_json(tmp_path, f"r:.sig.csd[{fmt(h)};{fmt(l)}]")
    k = 3 - 2 * np.sqrt(2)
    b = np.log(h / l) ** 2 + np.log(np.roll(h, 1) / np.roll(l, 1)) ** 2
    g = np.log(np.maximum(h, np.roll(h, 1)) / np.minimum(l, np.roll(l, 1))) ** 2
    a = (np.sqrt(2 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
    want = np.maximum(0, 2 * (np.exp(a) - 1) / (1 + np.exp(a)))
    want[0] = np.nan                                     # no previous day
    got = np.array([np.nan if v is None else v for v in got], dtype=float)
    assert (np.isnan(got) == np.isnan(want)).all()
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)], rtol=1e-12, atol=1e-15)


def test_overnight_plus_intraday_is_the_forward_log_return(tmp_path):
    r = q_json(tmp_path, f'raw:("DSSFFFFF";enlist ",") 0: `$"{ROOT}/data/sample/ohlcv.csv";'
                         'x:.sig.build[select date,sym,open,high,low,close,volume from raw;100;2019.01.01];'
                         'r:.ext.legcheck x')
    assert r < 1e-12


def test_aj_matches_pandas_merge_asof(tmp_path):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    panel = pd.DataFrame([(d, s) for d in dates[::5] for s in ("A", "B", "C")], columns=["date", "sym"])
    panel["close"] = rng.uniform(10, 100, len(panel))
    fund = pd.DataFrame({"sym": list("AABBC"), "fy": pd.to_datetime(
        ["2019-03-31", "2020-03-31", "2019-12-31", "2020-12-31", "2021-03-31"]),
        "ni": [1.0, 2.0, 3.0, 4.0, 5.0], "shares": [10.0, 10.0, 20.0, 20.0, 0.0]})
    fund["date"] = fund["fy"] + pd.Timedelta(days=90)
    d2q = lambda s: " ".join(pd.to_datetime(s).dt.strftime("%Y.%m.%d"))
    sy = lambda s: "`" + "`".join(s)
    fl = lambda s: " ".join(repr(float(v)) for v in s)
    body = (f"x:([] date:{d2q(panel.date)}; sym:{sy(panel.sym)}; close:{fl(panel.close)});"
            f"f:`sym`date xasc ([] sym:{sy(fund.sym)}; fy:{d2q(fund.fy)}; date:{d2q(fund.date)}; ni:{fl(fund.ni)}; shares:{fl(fund.shares)});"
            "r:exec ep from .ext.ep[x;f]")
    got = np.array([np.nan if v is None else v for v in q_json(tmp_path, body)], dtype=float)
    m = pd.merge_asof(panel.sort_values("date"), fund.sort_values("date"), on="date", by="sym",
                      direction="backward").set_index(["date", "sym"]).loc[list(zip(panel.date, panel.sym))]
    want = np.where((m.shares > 0) & (m.close > 0), m.ni / (m.shares * m.close), np.nan)
    assert (np.isnan(got) == np.isnan(want)).all()
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)], rtol=1e-14)


def test_adjust_removes_a_split_jump(tmp_path):
    body = ('t:([] date:2020.01.01+til 6; sym:6#`X; open:100 100 100 50 50 50f; high:100 100 100 50 50 50f;'
            ' low:100 100 100 50 50 50f; close:100 100 100 50 50 50f; volume:6#10f);'
            'ca:([] sym:enlist `X; date:enlist 2020.01.04; factor:enlist 0.5);'
            'a:.ext.adjust[t;ca]; r:`close`volume!(a`close;a`volume)')
    r = q_json(tmp_path, body)
    assert r["close"] == [50.0] * 6
    assert r["volume"] == [20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
