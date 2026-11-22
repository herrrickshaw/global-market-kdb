"""Offline tests for bench/bench_py.py (DuckDB side) and bench/run.py's orchestration.

The Polars half of bench_py needs polars, which is not a dependency of this repo, so only the
DuckDB path is exercised; bench/run.py is driven through a fake subprocess.
"""
from __future__ import annotations

import json
import runpy
import sys
import types
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
sys.path.insert(0, str(BENCH))
import bench_py  # noqa: E402
import run as bench_run  # noqa: E402


# ── bench_py ─────────────────────────────────────────────────────────────────
def make_warehouse(root):
    rng = np.random.default_rng(0)
    con = duckdb.connect()
    days = pd.bdate_range("2020-12-21", periods=20)

    def bars(syms):
        rows = []
        for s_i, s in enumerate(syms):
            px = 10 + s_i + np.cumsum(rng.normal(0, 0.1, len(days)))
            for d, c in zip(days, px):
                rows.append((d.date(), s, float(c), int(rng.integers(0, 500)) * (s != "ZZZ")))
        return pd.DataFrame(rows, columns=["Date", "Symbol", "Close", "Volume"])
    for mkt, syms in (("US", ["AAA", "BBB", "ZZZ"]), ("IN", ["INA", "INB"])):
        d = root / "ohlcv" / mkt
        d.mkdir(parents=True)
        df = bars(syms)
        con.register("df", df)
        con.execute(f"COPY (SELECT * FROM df) TO '{d / 'year=2020.parquet'}' (FORMAT PARQUET)")
        con.unregister("df")
        yield mkt, df
    f = root / "fundamentals_history"
    f.mkdir()
    fund = pd.DataFrame({"ticker": ["INA", "INA", "INB"],
                         "fy_end": pd.to_datetime(["2020-06-30", "2020-09-30", "2020-03-31"]).date,
                         "net_income": [1.0, 2.0, 3.0], "shares": [10.0, 10.0, 20.0]})
    con.register("fund", fund)
    con.execute(f"COPY (SELECT * FROM fund) TO '{f / 'IN_screener_only_backup.parquet'}' (FORMAT PARQUET)")


@pytest.fixture
def warehouse(tmp_path):
    frames = dict(make_warehouse(tmp_path))
    return tmp_path, frames


def jsonl(text):
    return [json.loads(x) for x in text.splitlines() if x.startswith("{")]


def test_duckdb_benchmark_emits_the_five_operations_with_correct_checks(warehouse, capsys):
    root, frames = warehouse
    bench_py.run_duckdb(str(root), 2, 1)
    rows = jsonl(capsys.readouterr().out)
    assert [r["op"] for r in rows] == ["load_us", "groupby_median", "window_by_sym", "rank_by_date", "asof_join_in"]
    assert {r["engine"] for r in rows} == {"duckdb-1t"} and {r["threads"] for r in rows} == {1}
    by = {r["op"]: r for r in rows}
    for r in rows:
        assert r["ms"] >= 0 and 0 <= r["min_ms"] <= r["ms"] + 1e-9

    us = frames["US"]
    assert by["load_us"]["check"] == len(us)
    ok = us[(us.Close > 0) & (us.Volume > 0)]
    assert by["groupby_median"]["check"] == ok.groupby(["Symbol", pd.to_datetime(ok.Date).dt.year]).ngroups
    s = us.assign(Date=pd.to_datetime(us.Date)).sort_values(["Symbol", "Date"])
    r1 = s.groupby("Symbol")["Close"].transform(lambda c: c / c.shift(1) - 1)
    am = r1.abs().groupby(s["Symbol"]).transform(lambda x: x.rolling(63, min_periods=1).mean())
    assert by["window_by_sym"]["check"] == pytest.approx(am.sum(), rel=1e-9)
    assert by["rank_by_date"]["check"] == len(us)
    # ASOF: each IN bar picks up the latest fundamentals row dated (fy_end + 90d) on or before it
    ind = frames["IN"].assign(Date=pd.to_datetime(frames["IN"].Date)).sort_values("Date")
    fund = pd.DataFrame({"sym": ["INA", "INA", "INB"],
                         "d": pd.to_datetime(["2020-06-30", "2020-09-30", "2020-03-31"]) + pd.Timedelta(days=90),
                         "ni": [1.0, 2.0, 3.0]}).sort_values("d")
    m = pd.merge_asof(ind, fund, left_on="Date", right_on="d", left_by="Symbol", right_by="sym")
    assert by["asof_join_in"]["check"] == int(m["ni"].notna().sum())


def test_threads_setting_is_reflected_in_engine_name(warehouse, capsys):
    bench_py.run_duckdb(str(warehouse[0]), 1, 3)
    assert {r["engine"] for r in jsonl(capsys.readouterr().out)} == {"duckdb-3t"}


def test_tm_returns_the_median_and_remembers_the_minimum(monkeypatch):
    clock = iter([0.0, 0.004, 1.0, 1.001, 2.0, 2.0025])             # durations 4, 1, 2.5 ms
    monkeypatch.setattr(bench_py.time, "perf_counter", lambda: next(clock))
    seq = iter(["a", "b", "c"])
    ms, value = bench_py.tm(lambda: next(seq), 3)
    assert ms == pytest.approx(2.5) and bench_py.MIN["last"] == pytest.approx(1.0) and value == "c"


def test_tm_even_number_of_reps_uses_mean_of_middle_pair(monkeypatch):
    clock = iter([0.0, 0.001, 1.0, 1.003])
    monkeypatch.setattr(bench_py.time, "perf_counter", lambda: next(clock))
    ms, _ = bench_py.tm(lambda: 1, 2)
    assert ms == pytest.approx(2.0) and bench_py.MIN["last"] == pytest.approx(1.0)


def test_emit_prints_one_json_line(capsys):
    bench_py.MIN["last"] = 1.5
    bench_py.emit("duckdb-4t", "load_us", 2.5, 99, 4)
    assert json.loads(capsys.readouterr().out) == {"engine": "duckdb-4t", "op": "load_us", "ms": 2.5,
                                                   "min_ms": 1.5, "check": 99, "threads": 4}


def test_script_entry_point_runs_duckdb_with_cli_arguments(warehouse, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bench_py.py", "duckdb", "--root", str(warehouse[0]), "--reps", "1",
                                      "--threads", "2"])
    runpy.run_path(str(BENCH / "bench_py.py"), run_name="__main__")
    rows = jsonl(capsys.readouterr().out)
    assert len(rows) == 5 and {r["engine"] for r in rows} == {"duckdb-2t"}


def test_script_entry_point_sets_polars_thread_cap_before_importing_polars(monkeypatch):
    class Boom(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError("polars used")
    monkeypatch.setitem(sys.modules, "polars", Boom("polars"))
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    monkeypatch.setattr(sys, "argv", ["bench_py.py", "polars", "--root", "/nowhere", "--threads", "3"])
    with pytest.raises(RuntimeError, match="polars used"):
        runpy.run_path(str(BENCH / "bench_py.py"), run_name="__main__")
    import os
    assert os.environ["POLARS_MAX_THREADS"] == "3"
    monkeypatch.delenv("POLARS_MAX_THREADS")


def test_script_rejects_unknown_engine(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench_py.py", "sqlite"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(BENCH / "bench_py.py"), run_name="__main__")
    assert e.value.code == 2


# ── bench/run.py ─────────────────────────────────────────────────────────────
def proc(rc=0, out="", err=""):
    return types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)


def test_lines_parses_only_json_lines(monkeypatch):
    seen = {}
    def fake(cmd, **kw):
        seen.update(cmd=cmd, kw=kw)
        return proc(0, 'banner\n{"op": "a", "ms": 1}\n  {"indented": 1}\n{"op": "b"}\n[1]\n')
    monkeypatch.setattr(bench_run.subprocess, "run", fake)
    assert bench_run.lines(["q", "x"]) == [{"op": "a", "ms": 1}, {"op": "b"}]
    assert seen["cmd"] == ["q", "x"] and seen["kw"] == {"cwd": bench_run.ROOT, "capture_output": True, "text": True}


def test_lines_exits_with_the_stderr_and_stdout_on_failure(monkeypatch):
    monkeypatch.setattr(bench_run.subprocess, "run", lambda cmd, **kw: proc(2, "OUT", "ERR"))
    with pytest.raises(SystemExit) as e:
        bench_run.lines(["/bin/q", "bench/bench.q"])
    assert str(e.value) == "/bin/q failed:\nERROUT"


def fake_polars(monkeypatch):
    mod = types.ModuleType("polars")
    mod.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, "polars", mod)


def make_runner(check_overrides=None, drop=()):
    """subprocess.run stand-in producing one JSON row per (engine, op) like bench.q / bench_py."""
    check_overrides = check_overrides or {}
    log = []
    ops = bench_run.OPS

    def row(engine, op, ms):
        check = check_overrides.get((engine, op), {"load_us": 1000, "groupby_median": 50, "window_by_sym": 7.5,
                                                   "rank_by_date": 1000, "asof_join_in": 900}[op.split("_peach")[0].split("_sorted")[0]
                                                   if op not in ops else op])
        return json.dumps({"engine": engine, "op": op, "ms": ms, "min_ms": ms / 2, "check": check})

    def fake(cmd, **kw):
        log.append((list(cmd), kw))
        if cmd[0] == "sysctl":
            return proc(0, {"machdep.cpu.brand_string": "Test CPU\n", "hw.ncpu": "8\n",
                            "hw.memsize": str(16 * 2**30) + "\n"}[cmd[2]])
        if "input" in kw:
            return proc(0, "4.1 2025.01.01\n")
        if "bench/bench.q" in cmd:
            rows = [row("q", op, 1000.0 + i) for i, op in enumerate(ops)]
            rows += [row("q", "load_us_peach", 700.0), row("q", "groupby_median_peach", 300.0),
                     row("q", "rank_by_date_sorted", 200.0)]
            return proc(0, "q banner\n" + "\n".join(rows) + "\n")
        eng = cmd[2]
        th = cmd[cmd.index("--threads") + 1]
        name = f"{eng}-{th}t"
        rows = [row(name, op, 10.0 * (j + 1)) for j, op in enumerate(ops) if (name, op) not in drop]
        return proc(0, "\n".join(rows) + "\n")
    return fake, log


@pytest.fixture
def bench_env(tmp_path, monkeypatch, capsys):
    (tmp_path / "bench").mkdir()
    monkeypatch.setattr(bench_run, "ROOT", tmp_path)
    fake_polars(monkeypatch)

    def go(runner, *argv):
        monkeypatch.setattr(bench_run.subprocess, "run", runner)
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        bench_run.main()
        return capsys.readouterr().out, (tmp_path / "bench" / "RESULTS.md").read_text()
    return go, tmp_path


def test_main_runs_every_engine_and_writes_results_md(bench_env):
    go, tmp = bench_env
    runner, log = make_runner()
    out, md = go(runner, "--reps", "3", "--root", "/wh", "--q", "/opt/q")
    cmds = [c for c, kw in log if "input" not in kw and c[0] != "sysctl"]
    assert cmds[0] == ["/opt/q", "bench/bench.q", "-root", "/wh", "-reps", "3", "-s", "4", "-q"]
    py = sys.executable
    assert cmds[1:] == [[py, "bench/bench_py.py", eng, "--root", "/wh", "--reps", "3", "--threads", str(th)]
                        for eng in ("duckdb", "polars") for th in (1, 4)]
    assert out == md.rstrip("\n") + "\n" or out.strip() == md.strip()
    lines = md.splitlines()
    assert lines[0] == "# Benchmark: q vs DuckDB vs Polars on the study's own operations"
    machine = lines[2]
    assert "Test CPU, 8 cores, 16 GB." in machine and "KDB-X 4.1 2025.01.01 (Community, `-s 4`)" in machine
    assert "Polars 9.9.9" in machine and f"DuckDB {duckdb.__version__}" in machine
    assert "Median (min) wall time in ms over 3 runs" in md
    assert md.endswith("\n") and not md.endswith("\n\n")


def table_rows(lines):
    """label -> list of cells (labels may themselves contain a '|')."""
    out = {}
    for label in bench_run.LABEL.values():
        pre = f"| {label} | "
        row = next(x for x in lines if x.startswith(pre))
        out[label] = [c.strip() for c in row[len(pre):].rstrip(" |").split(" | ")]
    return out


def test_main_table_cells_and_agreement(bench_env):
    go, _ = bench_env
    runner, _ = make_runner()
    _, md = go(runner)
    lines = md.splitlines()
    head = next(x for x in lines if x.startswith("| operation |"))
    assert head == ("| operation | q | q (peach / sorted) | duckdb-1t | duckdb-4t | polars-1t | polars-4t "
                    "| check agrees |")
    row = table_rows(lines)
    load = row[bench_run.LABEL["load_us"]]
    assert load == ["1,000 (500)", "700 (350)", "10 (5)", "10 (5)", "10 (5)", "10 (5)", "yes"]
    gm = row[bench_run.LABEL["groupby_median"]]
    assert gm[:2] == ["1,001 (500)", "300 (150)"]
    win = row[bench_run.LABEL["window_by_sym"]]
    assert win[1] == "–"                                          # no peach variant for this op
    assert row[bench_run.LABEL["rank_by_date"]][1] == "200 (100)"
    assert row[bench_run.LABEL["asof_join_in"]][1] == "–"
    assert all(r[-1] == "yes" for r in row.values())


def test_main_flags_engines_that_disagree_on_the_check_value(bench_env):
    go, _ = bench_env
    runner, _ = make_runner({("polars-4t", "groupby_median"): 51})
    _, md = go(runner)
    rows = {k: v[-1] for k, v in table_rows(md.splitlines()).items()}
    assert rows[bench_run.LABEL["groupby_median"]] == "NO"
    assert rows[bench_run.LABEL["load_us"]] == "yes"


def test_main_agreement_tolerance_is_relative_1e_6(bench_env):
    go, _ = bench_env
    ok, _ = make_runner({("duckdb-1t", "window_by_sym"): 7.5 * (1 + 5e-7)})
    _, md = go(ok)
    assert [x for x in md.splitlines() if bench_run.LABEL["window_by_sym"] in x][0].rstrip().endswith("| yes |")
    bad, _ = make_runner({("duckdb-1t", "window_by_sym"): 7.5 * (1 + 5e-6)})
    _, md = go(bad)
    assert [x for x in md.splitlines() if bench_run.LABEL["window_by_sym"] in x][0].rstrip().endswith("| NO |")


def test_main_shows_na_for_engines_that_did_not_report(bench_env):
    go, _ = bench_env
    runner, _ = make_runner(drop={("polars-1t", "rank_by_date")})
    _, md = go(runner)
    cells = table_rows(md.splitlines())[bench_run.LABEL["rank_by_date"]]
    assert cells[4] == "n/a" and cells[-1] == "yes"


def test_main_defaults_use_home_warehouse_and_five_reps(bench_env, monkeypatch):
    go, _ = bench_env
    runner, log = make_runner()
    monkeypatch.setenv("HOME", "/h")
    go(runner)
    first = log[0][0]
    assert first == [str(Path("/h/.kx/bin/q")), "bench/bench.q", "-root", str(Path("/h/.cache/market-warehouse")),
                     "-reps", "5", "-s", "4", "-q"]
