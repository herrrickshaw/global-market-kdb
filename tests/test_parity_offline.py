"""Offline tests for python/parity.py: no q binary, no subprocess, no warehouse.

``subprocess.run`` is replaced by a recorder that writes the CSVs q would have written, and
``find_q`` / ``study_py.run`` / ``extract_sample.query`` are faked where main() needs them.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import parity  # noqa: E402
import study_py  # noqa: E402


# ── find_q ───────────────────────────────────────────────────────────────────
@pytest.fixture
def q_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("Q", raising=False)
    monkeypatch.setattr(parity.shutil, "which", lambda name: None)
    return tmp_path


def touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    return str(p)


def test_find_q_prefers_env_then_path_then_home(q_env, monkeypatch):
    assert parity.find_q() is None
    home_q = touch(q_env / "home" / ".kx" / "bin" / "q")
    assert parity.find_q() == home_q
    path_q = touch(q_env / "bin" / "q")
    monkeypatch.setattr(parity.shutil, "which", lambda name: path_q if name == "q" else None)
    assert parity.find_q() == path_q
    env_q = touch(q_env / "elsewhere" / "q")
    monkeypatch.setenv("Q", env_q)
    assert parity.find_q() == env_q


def test_find_q_skips_candidates_that_do_not_exist(q_env, monkeypatch):
    home_q = touch(q_env / "home" / ".kx" / "bin" / "q")
    monkeypatch.setenv("Q", str(q_env / "missing" / "q"))
    monkeypatch.setattr(parity.shutil, "which", lambda name: str(q_env / "gone" / "q"))
    assert parity.find_q() == home_q
    monkeypatch.setenv("Q", "")
    assert parity.find_q() == home_q


# ── _norm ────────────────────────────────────────────────────────────────────
def test_norm_converts_date_strings_with_dashes_or_dots():
    df = pd.DataFrame({"a": ["2020-01-02", "2020-02-03"], "b": ["2020.01.02", "2021.12.31"],
                       "n": [1.5, 2.5], "lab": ["x", "y"]})
    out = parity._norm(df)
    assert out["a"].tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-02-03")]
    assert out["b"].tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2021-12-31")]
    assert out["n"].tolist() == [1.5, 2.5] and out["lab"].tolist() == ["x", "y"]


def test_norm_leaves_partly_date_like_columns_alone():
    df = pd.DataFrame({"a": ["2020-01-02", "n/a"], "b": ["2020-1-2", "2020-01-02"]})
    out = parity._norm(df)
    assert out["a"].tolist() == ["2020-01-02", "n/a"]
    assert out["b"].tolist() == ["2020-1-2", "2020-01-02"]


def test_norm_true_false_strings_become_bool_even_when_one_sided():
    df = pd.DataFrame({"f": ["True", "False", "True"], "all_true": ["True"] * 3, "mix": ["True", "yes", "False"]})
    out = parity._norm(df)
    assert out["f"].tolist() == [True, False, True] and out["f"].dtype == bool
    assert out["all_true"].tolist() == [True] * 3
    assert out["mix"].tolist() == ["True", "yes", "False"]


def test_norm_does_not_touch_numeric_or_bool_and_does_not_mutate_input():
    df = pd.DataFrame({"i": [1, 2], "b": [True, False], "d": ["2020-01-02", "2020-01-03"]})
    snapshot = df.copy()
    out = parity._norm(df)
    pd.testing.assert_frame_equal(df, snapshot)                      # input left as it was
    assert out["i"].tolist() == [1, 2] and out["b"].tolist() == [True, False]
    assert out is not df


# ── compare ──────────────────────────────────────────────────────────────────
def ic_df(**over):
    d = {"market": ["US", "US", "IN"], "signal": ["rev5", "mom21", "rev5"], "h": [5, 10, 5],
         "mean_ic": [0.01, 0.02, 0.03], "hit": [0.5, 0.6, 0.7]}
    d.update(over)
    return pd.DataFrame(d)


def cmp(q, py, name="ic_table", atol=1e-10, rtol=1e-8):
    return parity.compare(name, q, py, atol, rtol)


def test_compare_identical_frames_pass():
    r = cmp(ic_df(), ic_df())
    assert r == {"table": "ic_table", "rows": 3, "max_abs_diff": 0.0, "mismatches": 0, "pass": True, "worst": ""}


def test_compare_empty_and_missing_sides():
    empty = ic_df().iloc[0:0]
    for q, py in [(None, None), (None, empty), (empty, None), (empty, empty)]:
        assert cmp(q, py)["pass"] is True and cmp(q, py)["mismatches"] == 0
    r = cmp(None, ic_df())
    assert r["pass"] is False and r["mismatches"] == 1 and r["rows"] == 0
    r = cmp(ic_df(), None)
    assert r["pass"] is False and r["mismatches"] == 1
    r = cmp(ic_df(), empty)
    assert r["pass"] is False and r["mismatches"] == 1                # q has rows, python has none
    assert cmp(empty, ic_df())["pass"] is False


def test_compare_tolerance_boundary_is_inclusive():
    py = ic_df(mean_ic=[1.0, 2.0, 4.0])
    ok = ic_df(mean_ic=[1.5, 2.0, 4.0])
    assert cmp(ok, py, atol=0.5, rtol=0.0)["pass"] is True            # |diff| == atol is allowed
    assert cmp(ic_df(mean_ic=[1.5000001, 2.0, 4.0]), py, atol=0.5, rtol=0.0)["pass"] is False
    # relative part scales with |py|: tol = atol + rtol * |py|
    r = cmp(ic_df(mean_ic=[1.0, 2.0, 5.0]), py, atol=0.0, rtol=0.25)
    assert r["pass"] is True                                           # 4 * 0.25 == 1 == |5 - 4|
    r = cmp(ic_df(mean_ic=[1.0, 2.0, 5.0001]), py, atol=0.0, rtol=0.25)
    assert r["pass"] is False and r["mismatches"] == 1


def test_compare_reports_worst_column_and_max_diff():
    q = ic_df(mean_ic=[0.01, 0.02, 0.03], hit=[0.5, 0.7, 0.7])
    r = cmp(q, ic_df(), atol=1e-12, rtol=0)
    assert r["mismatches"] == 1 and r["pass"] is False
    assert r["worst"] == "hit" and r["max_abs_diff"] == pytest.approx(0.1)
    r2 = cmp(ic_df(mean_ic=[0.01, 0.5, 0.03], hit=[0.5, 0.7, 0.7]), ic_df(), atol=1e-12, rtol=0)
    assert r2["worst"] == "mean_ic" and r2["max_abs_diff"] == pytest.approx(0.48)


def test_compare_null_pattern_must_match():
    py = ic_df(mean_ic=[np.nan, 0.02, 0.03])
    assert cmp(ic_df(mean_ic=[np.nan, 0.02, 0.03]), py)["pass"] is True
    r = cmp(ic_df(mean_ic=[0.0, 0.02, 0.03]), py)
    assert r["pass"] is False and r["mismatches"] == 1
    r = cmp(ic_df(mean_ic=[0.01, np.nan, 0.03]), ic_df())
    assert r["pass"] is False and r["mismatches"] == 1


def test_compare_row_set_differences_count_from_both_sides():
    q = ic_df().iloc[:2]
    r = cmp(q, ic_df())
    assert r["rows"] == 3 and r["mismatches"] >= 1 and r["pass"] is False
    extra = pd.concat([ic_df(), ic_df().iloc[:1].assign(h=63)], ignore_index=True)
    r = cmp(extra, ic_df())
    assert r["rows"] == 4 and r["pass"] is False


def test_compare_keys_are_matched_after_type_normalisation():
    q = ic_df(h=["5", "10", "5"])                                     # read back as strings
    assert cmp(q, ic_df())["pass"] is True
    q = ic_df()
    assert cmp(q.iloc[::-1].reset_index(drop=True), ic_df())["pass"] is True     # row order is irrelevant


def test_compare_integer_keys_survive_float_formatting():
    q = ic_df(h=[5.0, 10.0, 5.0])                                       # read_csv gives floats when a key column has NaN
    assert cmp(q, ic_df())["pass"] is True


def test_compare_date_keys_accept_q_dotted_dates():
    py = pd.DataFrame({"market": ["US"] * 2, "signal": ["a"] * 2, "h": [10, 10],
                       "date": pd.to_datetime(["2020-01-03", "2020-01-10"]), "ic": [0.1, 0.2]})
    q = py.assign(date=["2020.01.03", "2020.01.10"])
    assert parity.compare("ic_series", q, py, 1e-10, 1e-8)["pass"] is True
    q2 = py.assign(date=["2020.01.03", "2020.01.17"])
    assert parity.compare("ic_series", q2, py, 1e-10, 1e-8)["pass"] is False


def test_compare_fold_key_cast_to_int():
    py = pd.DataFrame({"market": ["US"], "fold": [1], "oos": [0.5]})
    q = pd.DataFrame({"market": ["US"], "fold": ["1"], "oos": [0.5]})
    assert parity.compare("walk_forward_10d", q, py, 1e-10, 1e-8)["pass"] is True


def test_compare_label_and_bool_columns_compare_exactly():
    py = pd.DataFrame({"market": ["US", "US"], "signal": ["a", "b"], "status": ["HEALTHY", "NOISE"],
                       "flag": [True, False]})
    assert parity.compare("health", py.copy(), py, 1e-10, 1e-8)["pass"] is True
    bad = py.assign(status=["HEALTHY", "FLIPPED"])
    r = parity.compare("health", bad, py, 1e-10, 1e-8)
    assert r["pass"] is False and r["mismatches"] == 1
    # q writes booleans as 0/1, "true", or "True"
    for spelled in ([1, 0], ["true", "false"], ["True", "False"]):
        assert parity.compare("health", py.assign(flag=spelled), py, 1e-10, 1e-8)["pass"] is True, spelled
    r = parity.compare("health", py.assign(flag=[0, 1]), py, 1e-10, 1e-8)
    assert r["pass"] is False and r["mismatches"] == 2


def test_compare_numeric_column_with_unparseable_q_cells_counts_as_mismatch():
    q = ic_df(mean_ic=["0.01", "oops", "0.03"])
    r = cmp(q, ic_df())
    assert r["pass"] is False and r["mismatches"] == 1
    assert cmp(ic_df(mean_ic=["0.01", "0.02", "0.03"]), ic_df())["pass"] is True


def test_compare_exempt_cells_only_check_the_null_pattern():
    py = pd.DataFrame({"market": ["US", "US"], "signal": ["lowvol", "rev5"], "h": [10, 10],
                       "neutral_ic": [1e-16, 0.01], "neutral_t": [0.3, 0.5], "mean_ic": [0.1, 0.2]})
    noisy = py.assign(neutral_ic=[-3e-16, 0.01], neutral_t=[7.0, 0.5])
    r = parity.compare("ic_table", noisy, py, 1e-10, 1e-8)
    assert r["pass"] is True and r["max_abs_diff"] == 0.0 and r["worst"] == ""
    nullmiss = py.assign(neutral_ic=[np.nan, 0.01])
    assert parity.compare("ic_table", nullmiss, py, 1e-10, 1e-8)["pass"] is False
    # the same tolerance does bite on a non-exempt signal and on non-exempt columns
    off = py.assign(neutral_ic=[1e-16, 0.5])
    assert parity.compare("ic_table", off, py, 1e-10, 1e-8)["pass"] is False
    off2 = py.assign(mean_ic=[0.9, 0.2])
    assert parity.compare("ic_table", off2, py, 1e-10, 1e-8)["pass"] is False
    # the exemption is specific to the ic_table table
    assert parity.compare("costs", noisy, py, 1e-10, 1e-8)["pass"] is False


# ── run_q ────────────────────────────────────────────────────────────────────
class FakeRun:
    """Recorder standing in for subprocess.run; emulates q writing the result CSVs."""

    def __init__(self, tables=None, fail_on=None, stdout="loaded 42 rows\nmarket x\nnoise", stderr=""):
        self.calls, self.tables = [], tables if tables is not None else {}
        self.fail_on, self.stdout, self.stderr = fail_on, stdout, stderr

    def __call__(self, cmd, **kw):
        self.calls.append((list(cmd), kw))
        script = cmd[1]
        if script == self.fail_on:
            return types.SimpleNamespace(returncode=3, stdout="out-text", stderr="err-text")
        if script == "q/run.q":
            out = Path(cmd[cmd.index("-out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            for k in parity.KEYS:
                t = self.tables.get(k)
                if t is None:
                    (out / f"{k}.csv").write_text("\n")                # q writes a blank file for no rows
                else:
                    t.to_csv(out / f"{k}.csv", index=False)
        return types.SimpleNamespace(returncode=0, stdout=self.stdout, stderr=self.stderr)


def test_run_q_csv_commands_and_result_collection(tmp_path, monkeypatch, capsys):
    fake = FakeRun(tables={"ic_table": ic_df(), "health": pd.DataFrame({"market": ["US"], "signal": ["a"]})})
    monkeypatch.setattr(parity.subprocess, "run", fake)
    hdb = tmp_path / "w" / "db"
    hdb.mkdir(parents=True)
    (hdb / "stale").write_text("x")                                    # a previous HDB must be wiped
    res = parity.run_q("/bin/q", Path("/data/s.csv"), 77, "2019-06-01", tmp_path / "w")
    (load, kw1), (run, kw2) = fake.calls
    assert load == ["/bin/q", "q/load.q", "-csv", "/data/s.csv", "-hdb", str(hdb), "-q"]
    assert run == ["/bin/q", "q/run.q", "-hdb", str(hdb), "-univ", "77", "-start", "2019.06.01",
                   "-out", str(tmp_path / "w" / "q"), "-q"]
    for kw in (kw1, kw2):
        assert kw == {"cwd": parity.ROOT, "capture_output": True, "text": True}
    assert not (hdb / "stale").exists()
    assert set(res) == {"ic_table", "health"}                          # empty files are skipped
    pd.testing.assert_frame_equal(res["ic_table"], ic_df())
    # q's progress output is echoed up to the first results table, with one trailing newline
    assert capsys.readouterr().out == "loaded 42 rows\nloaded 42 rows\n"


def test_run_q_parquet_commands(tmp_path, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(parity.subprocess, "run", fake)
    pq = {"root": "/wh/ohlcv", "mkts": ["US", "IN"], "from": "2016-06-01", "to": "2022-12-31", "names": 200}
    assert parity.run_q("q", None, 100, "2019-01-01", tmp_path, pq) == {}
    load = fake.calls[0][0]
    assert load == ["q", "q/pqload.q", "-root", "/wh/ohlcv", "-mkts", "US", "IN", "-from", "2016.06.01",
                    "-to", "2022.12.31", "-names", "200", "-hdb", str(tmp_path / "db"), "-q"]
    assert fake.calls[1][0][1] == "q/run.q" and fake.calls[1][0][fake.calls[1][0].index("-start") + 1] == "2019.01.01"


@pytest.mark.parametrize("fail_on,first_only", [("q/load.q", True), ("q/run.q", False)])
def test_run_q_failure_exits_with_q_output(tmp_path, monkeypatch, fail_on, first_only):
    fake = FakeRun(fail_on=fail_on)
    monkeypatch.setattr(parity.subprocess, "run", fake)
    with pytest.raises(SystemExit) as e:
        parity.run_q("q", Path("a.csv"), 10, "2019-01-01", tmp_path)
    assert str(e.value) == f"q {fail_on} failed (3):\nerr-textout-text"
    assert len(fake.calls) == (1 if first_only else 2)                 # a failed load stops the study


# ── main ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    state = types.SimpleNamespace(run_q=[], py=[], q_tables={}, py_tables={})
    monkeypatch.setattr(parity, "find_q", lambda: "/fake/q")

    def fake_run_q(q, csv, n, start, work, pq=None):
        state.run_q.append((q, csv, n, start, work, pq))
        return dict(state.q_tables)

    def fake_py(src, n, start):
        state.py.append((src, n, start))
        return dict(state.py_tables)
    monkeypatch.setattr(parity, "run_q", fake_run_q)
    monkeypatch.setattr(study_py, "run", fake_py)

    def go(*argv):
        monkeypatch.setattr(sys, "argv", ["parity.py", *argv])
        try:
            parity.main()
            code = 0
        except SystemExit as e:
            code = e.code
        return code, capsys.readouterr().out
    state.go, state.tmp = go, tmp_path
    return state


def test_main_without_q_exits_with_install_hint(cli, monkeypatch):
    monkeypatch.setattr(parity, "find_q", lambda: None)
    code, _ = cli.go("--work", str(cli.tmp / "w"))
    assert code == "q not found: install KDB-X (see README) or set Q=/path/to/q"


def test_main_csv_mode_pass(cli):
    csv = cli.tmp / "sample.csv"
    code, out = cli.go("--csv", str(csv), "--work", str(cli.tmp / "w" / "deep"), "--n", "50",
                       "--start", "2020-02-01", "--atol", "1e-6", "--rtol", "1e-4")
    assert code == 0
    assert cli.run_q == [("/fake/q", csv.resolve(), 50, "2020-02-01", (cli.tmp / "w" / "deep").resolve(), None)]
    assert cli.py == [(csv.resolve(), 50, "2020-02-01")]
    assert (cli.tmp / "w" / "deep").is_dir()                          # created, parents included
    lines = out.splitlines()
    assert "parity on sample.csv: n=50, start=2020-02-01, atol=1e-06, rtol=0.0001" in lines
    assert lines[-1] == "PASS"
    assert "exempt (null pattern only): ic_table lowvol ['neutral_ic', 'neutral_t'] - degenerate upstream value, see README" in lines
    header = next(x for x in lines if x.startswith("            table") or x.lstrip().startswith("table"))
    assert header.split() == ["table", "rows", "max_abs_diff", "mismatches", "pass", "worst"]
    for k in parity.KEYS:
        assert any(x.split()[:1] == [k] for x in lines), k


def test_main_defaults(cli):
    code, out = cli.go("--work", str(cli.tmp / "w"))
    assert code == 0
    q, csv, n, start, work, pq = cli.run_q[0]
    assert csv == (parity.ROOT / "data/sample/ohlcv.csv").resolve() and n == 100 and start == "2019-01-01"
    assert "atol=1e-10, rtol=1e-08" in out and "parity on ohlcv.csv: n=100" in out


def test_main_fail_exit_code_and_fail_banner(cli):
    cli.py_tables["ic_table"] = ic_df()                                # q produced nothing for it
    code, out = cli.go("--work", str(cli.tmp / "w"))
    assert code == 1
    assert out.splitlines()[-2:][0].startswith("exempt") and out.splitlines()[-1] == "FAIL"
    row = next(x for x in out.splitlines() if x.split()[:1] == ["ic_table"])
    assert row.split()[3] == "1" and row.split()[4] == "False"


def test_main_one_failing_table_among_passing_ones(cli):
    cli.q_tables = {"ic_table": ic_df(), "costs": ic_df()}
    cli.py_tables = {"ic_table": ic_df(), "costs": ic_df(mean_ic=[9.0, 0.02, 0.03])}
    code, out = cli.go("--work", str(cli.tmp / "w"))
    assert code == 1 and out.splitlines()[-1] == "FAIL"
    rows = {x.split()[0]: x.split() for x in out.splitlines() if x.split()[:1] and x.split()[0] in parity.KEYS}
    assert rows["ic_table"][4] == "True" and rows["costs"][4] == "False"


def test_main_parquet_mode(cli, monkeypatch):
    import extract_sample
    seen = []

    def fake_query(root, mkt, names, d0, d1):
        seen.append((root, mkt, names, d0, d1))
        return pd.DataFrame({"date": [pd.Timestamp("2020-01-02")], "mkt": [mkt], "sym": ["A"],
                             "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
    monkeypatch.setattr(extract_sample, "query", fake_query)
    monkeypatch.setenv("HOME", str(cli.tmp))
    code, out = cli.go("--parquet", "~/wh", "--mkts", "US", "JP", "--from", "2017-01-01", "--to", "2019-12-31",
                       "--names", "33", "--work", str(cli.tmp / "w"), "--n", "20")
    assert code == 0
    root = str(cli.tmp / "wh")
    assert seen == [(root, "US", 33, "2017-01-01", "2019-12-31"), (root, "JP", 33, "2017-01-01", "2019-12-31")]
    q, csv, n, start, work, pq = cli.run_q[0]
    assert csv is None and pq == {"root": root, "mkts": ["US", "JP"], "from": "2017-01-01",
                                  "to": "2019-12-31", "names": 33}
    src, n2, start2 = cli.py[0]
    assert isinstance(src, pd.DataFrame) and list(src["mkt"]) == ["US", "JP"] and n2 == 20
    assert f"parity on parquet {root} US JP 2017-01-01..2019-12-31, top 33: n=20, start=2019-01-01" in out


def test_main_parquet_defaults(cli, monkeypatch):
    import extract_sample
    calls = []
    monkeypatch.setattr(extract_sample, "query", lambda *a: calls.append(a) or pd.DataFrame())
    code, out = cli.go("--parquet", "/wh", "--work", str(cli.tmp / "w"))
    assert code == 0 and calls == [("/wh", "US", 200, "2016-06-01", "2022-12-31")]
    assert cli.run_q[0][5] == {"root": "/wh", "mkts": ["US"], "from": "2016-06-01", "to": "2022-12-31", "names": 200}
