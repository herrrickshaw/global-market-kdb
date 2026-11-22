"""Offline tests for python/pykx_parity.py with a fake ``pykx`` module (no licence, no q)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import parity  # noqa: E402
import pykx_parity  # noqa: E402
import study_py  # noqa: E402


class FakeTable:
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def pd(self):
        return self.df


class FakeKx(types.ModuleType):
    def __init__(self, licensed=True, results=None):
        super().__init__("pykx")
        self.licensed = licensed
        self.results = results if results is not None else {}
        self.calls, self.toq_args = [], []

    def toq(self, df):
        self.toq_args.append(df)
        return ("toq", len(df))

    def q(self, code, *args):
        self.calls.append((code, args))
        if code == ".study.run":
            return {k: FakeTable(self.results.get(k, pd.DataFrame())) for k in parity.KEYS}
        return code


def sample_csv(path):
    pd.DataFrame({
        "date": ["2020-01-02", "2020-01-03", "2020-01-03", "2020-01-06"],
        "mkt": ["SYN"] * 4, "sym": ["A", "A", "B", "A"],
        "open": [1.0, 1.0, 2.0, 1.0], "high": [1.1, 1.1, 2.2, 1.1], "low": [0.9, 0.9, 1.8, 0.9],
        "close": [1.0, 1.0, 2.0, 0.0], "volume": [100, 0, 50, 70]}).to_csv(path, index=False)
    return path


@pytest.fixture
def harness(tmp_path, monkeypatch, capsys):
    st = types.SimpleNamespace(kx=FakeKx(), py_calls=[], py_tables={})
    monkeypatch.setitem(sys.modules, "pykx", st.kx)
    monkeypatch.setattr(study_py, "run",
                        lambda src, n, start: st.py_calls.append((src, n, start)) or dict(st.py_tables))
    csv = sample_csv(tmp_path / "s.csv")

    def go(*argv):
        monkeypatch.setattr(sys, "argv", ["pykx_parity.py", "--csv", str(csv), *argv])
        try:
            pykx_parity.main()
            code = 0
        except SystemExit as e:
            code = e.code
        return code, capsys.readouterr().out
    st.go, st.csv = go, csv
    return st


def test_unlicensed_pykx_is_refused_before_loading_anything(harness):
    harness.kx.licensed = False
    code, out = harness.go()
    assert code == "PyKX is unlicensed and cannot run q code; install a KDB-X licence (see README)"
    assert harness.kx.calls == [] and harness.py_calls == []


def test_loads_the_three_q_files_in_order_then_runs_the_study(harness):
    code, out = harness.go("--n", "40", "--start", "2020-02-01")
    assert code == 0
    codes = [c for c, _ in harness.kx.calls]
    root = parity.ROOT
    assert codes[:3] == [f'system "l {root}/q/{f}.q"' for f in ("stats", "signals", "study")]
    assert codes[3] == '{update `date$date from x}'
    assert codes[4] == '"D"$"2020-02-01"'
    run_code, run_args = harness.kx.calls[-1]
    assert run_code == ".study.run" and run_args == ('{update `date$date from x}', 40, '"D"$"2020-02-01"')
    assert harness.kx.calls[3] == ('{update `date$date from x}', (("toq", 3),))
    assert harness.py_calls == [(str(harness.csv), 40, "2020-02-01")]


def test_table_handed_to_q_is_filtered_typed_and_ordered(harness):
    harness.go()
    (df,) = harness.kx.toq_args
    assert list(df.columns) == ["date", "mkt", "sym", "open", "high", "low", "close", "volume"]
    assert len(df) == 3 and (df.close > 0).all()                    # the close == 0 row is dropped
    assert df.volume.dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(df.date)


def test_defaults(harness, monkeypatch):
    harness.go()
    assert harness.py_calls[0][1:] == (100, "2019-01-01")
    code_run = harness.kx.calls[-1]
    assert code_run[1][1] == 100


def test_passes_when_both_sides_agree_and_prints_the_exemption(harness):
    ic = pd.DataFrame({"market": ["US"], "signal": ["a"], "h": [5], "mean_ic": [0.1]})
    harness.kx.results = {"ic_table": ic}
    harness.py_tables = {"ic_table": ic.copy()}
    code, out = harness.go()
    assert code == 0 and out.splitlines()[-1] == "PASS"
    assert "exempt (null pattern only): ic_table lowvol ['neutral_ic', 'neutral_t']" in out.splitlines()
    assert any(x.split()[:1] == ["ic_table"] and x.split()[1] == "1" for x in out.splitlines())


def test_fails_with_exit_code_1_on_disagreement(harness):
    ic = pd.DataFrame({"market": ["US"], "signal": ["a"], "h": [5], "mean_ic": [0.1]})
    harness.kx.results = {"ic_table": ic.assign(mean_ic=0.5)}
    harness.py_tables = {"ic_table": ic}
    code, out = harness.go()
    assert code == 1 and out.splitlines()[-1] == "FAIL"


def test_empty_q_tables_are_treated_as_missing(harness):
    harness.py_tables = {"costs": pd.DataFrame({"market": ["US"], "signal": ["a"], "h": [5], "x": [1.0]})}
    code, out = harness.go()                                           # q returned zero rows for every table
    assert code == 1
    row = next(x for x in out.splitlines() if x.split()[:1] == ["costs"])
    assert row.split()[3] == "1"                                       # one mismatch: python has rows, q has none


def test_tolerances_are_forwarded(harness):
    ic = pd.DataFrame({"market": ["US"], "signal": ["a"], "h": [5], "mean_ic": [1.0]})
    harness.kx.results = {"ic_table": ic.assign(mean_ic=1.5)}
    harness.py_tables = {"ic_table": ic}
    assert harness.go("--atol", "0.5", "--rtol", "0")[0] == 0
    assert harness.go("--atol", "0.4999", "--rtol", "0")[0] == 1
