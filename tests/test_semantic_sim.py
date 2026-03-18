import copy
import unittest
from pathlib import Path

from compiler.ast_to_hir import lower_source
from compiler.hir_to_lir import lower_func
from compiler.sim_runtime import (
    PrimitiveModel,
    SimulationError,
    SimulationHarness,
    compare_simulation_results,
    run_all,
    run_hir,
    run_lir,
    run_python,
)


ROOT = Path(__file__).resolve().parents[1]
LU_CORE = (ROOT / "examples" / "lu_core.dsl.py").read_text()


def add1_comb(tb, v):
    return v + 1


def lt_comb(tb, a, b):
    return a < b


def pick_comb(tb, a, b):
    return a + b


def is_even_comb(tb, v):
    return (v & 1) == 0


def load(tb, i):
    return tb.state["mem"][i]


def record(tb, i, v):
    tb.state["seen"].append((i, v))


def push(tb, v):
    tb.state["out"].append(v)


def fetch_A(tb, i, j):
    return tb.state["A"][i][j]


def fetch_LU(tb, i, j):
    return tb.state["LU"][i][j]


def store_LU(tb, i, j, v):
    tb.state["LU"][i][j] = v


def neg_comb(tb, v):
    return -v


def fma(tb, a, b, c):
    return (a * b) + c


def div(tb, a, b):
    return a / b


class SemanticSimulationTests(unittest.TestCase):
    def test_simple_arithmetic_kernel_matches_across_python_hir_lir(self) -> None:
        source = '''
def accum(par_n):
    u8_i = 0
    u8_sum = 0
    for u8_i in range(par_n):
        u8_sum = u8_sum + u8_i
    return u8_sum
'''
        harness = SimulationHarness(params={"par_n": 5}, primitives={}, initial_state={"tag": "unchanged"})
        report = run_all(source, harness)

        self.assertTrue(report.ok, report.mismatches)
        self.assertEqual(report.python_result.return_value, 10)
        self.assertEqual(report.python_result.locals, {"u8_i": 4, "u8_sum": 10})
        self.assertEqual(report.python_result.final_state, {"tag": "unchanged"})
        self.assertEqual(report.python_result.trace, [])
        self.assertEqual(harness.initial_state, {"tag": "unchanged"})

    def test_comb_primitives_work_in_assignment_and_conditional_expressions(self) -> None:
        source = '''
def comb_kernel(par_n):
    u8_i = 0
    u8_out = 0
    for u8_i in range(par_n):
        u8_out = add1_comb(v=u8_out)
        u8_out = pick_comb(a=u8_out, b=u8_i) if lt_comb(a=u8_i, b=par_n) else 0
    return u8_out
'''
        harness = SimulationHarness(
            params={"par_n": 4},
            primitives={
                "add1_comb": PrimitiveModel(name="add1_comb", impl=add1_comb),
                "lt_comb": PrimitiveModel(name="lt_comb", impl=lt_comb),
                "pick_comb": PrimitiveModel(name="pick_comb", impl=pick_comb),
            },
            initial_state={},
        )
        report = run_all(source, harness)

        self.assertTrue(report.ok, report.mismatches)
        self.assertEqual(report.python_result.return_value, 10)
        self.assertEqual(
            [entry.name for entry in report.python_result.trace],
            [
                "add1_comb",
                "lt_comb",
                "pick_comb",
                "add1_comb",
                "lt_comb",
                "pick_comb",
                "add1_comb",
                "lt_comb",
                "pick_comb",
                "add1_comb",
                "lt_comb",
                "pick_comb",
            ],
        )
        self.assertTrue(all(entry.kind == "comb" for entry in report.python_result.trace))

    def test_blocking_assignment_and_side_effect_calls_preserve_state_and_latency(self) -> None:
        source = '''
def io_kernel(par_n):
    u8_i = 0
    u8_x = 0
    for u8_i in range(par_n):
        u8_x = load(i=u8_i)
        record(i=u8_i, v=u8_x)
    return u8_x
'''
        initial_state = {"mem": [7, 11, 13], "seen": []}
        harness = SimulationHarness(
            params={"par_n": 3},
            primitives={
                "load": PrimitiveModel(name="load", impl=load, latency=3),
                "record": PrimitiveModel(name="record", impl=record, latency=2),
            },
            initial_state=initial_state,
        )
        report = run_all(source, harness)

        self.assertTrue(report.ok, report.mismatches)
        self.assertEqual(report.python_result.return_value, 13)
        self.assertEqual(
            report.python_result.final_state["seen"],
            [(0, 7), (1, 11), (2, 13)],
        )
        self.assertEqual(report.hir_result.stats["latency_cycles"], 15)
        self.assertEqual(report.lir_result.stats["latency_cycles"], 15)
        self.assertEqual(initial_state, {"mem": [7, 11, 13], "seen": []})

    def test_if_while_and_for_stateful_control_flow_matches(self) -> None:
        source = '''
def control_kernel(par_n):
    u8_i = 0
    u8_acc = 0
    while u8_i < par_n:
        if is_even_comb(v=u8_i):
            push(v=u8_i)
            u8_acc = u8_acc + u8_i
        else:
            push(v=0)
        u8_i = u8_i + 1
    return u8_acc
'''
        harness = SimulationHarness(
            params={"par_n": 5},
            primitives={
                "is_even_comb": PrimitiveModel(name="is_even_comb", impl=is_even_comb),
                "push": PrimitiveModel(name="push", impl=push, latency=2),
            },
            initial_state={"out": []},
        )
        report = run_all(source, harness)

        self.assertTrue(report.ok, report.mismatches)
        self.assertEqual(report.python_result.return_value, 6)
        self.assertEqual(report.python_result.final_state["out"], [0, 0, 2, 0, 4])

    def test_trace_mismatch_detection_reports_first_difference(self) -> None:
        source = '''
def tracey(par_n):
    u8_i = 0
    u8_out = 0
    u8_out = add1_comb(v=u8_i)
    return u8_out
'''
        good_harness = SimulationHarness(
            params={"par_n": 1},
            primitives={"add1_comb": PrimitiveModel(name="add1_comb", impl=add1_comb)},
            initial_state={},
        )
        bad_harness = SimulationHarness(
            params={"par_n": 1},
            primitives={"add1_comb": PrimitiveModel(name="add1_comb", impl=lambda tb, v: v + 2)},
            initial_state={},
        )
        python_result = run_python(source, good_harness)
        hir_result = run_hir(lower_source(source), bad_harness)

        mismatches = compare_simulation_results("python", python_result, "hir", hir_result)
        self.assertTrue(any("return_value differs" in item for item in mismatches))
        self.assertTrue(any("trace differs" in item for item in mismatches))

    def test_state_mismatch_detection_reports_final_state_difference(self) -> None:
        source = '''
def stateful(par_n):
    u8_i = 0
    u8_x = 0
    u8_x = load(i=u8_i)
    record(i=u8_i, v=u8_x)
    return u8_x
'''
        python_harness = SimulationHarness(
            params={"par_n": 1},
            primitives={
                "load": PrimitiveModel(name="load", impl=load, latency=1),
                "record": PrimitiveModel(name="record", impl=record, latency=1),
            },
            initial_state={"mem": [5], "seen": []},
        )
        lir_harness = SimulationHarness(
            params={"par_n": 1},
            primitives={
                "load": PrimitiveModel(name="load", impl=load, latency=1),
                "record": PrimitiveModel(name="record", impl=record, latency=1),
            },
            initial_state={"mem": [9], "seen": []},
        )
        python_result = run_python(source, python_harness)
        lir_result = run_lir(lower_func(lower_source(source)), lir_harness)

        mismatches = compare_simulation_results("python", python_result, "lir", lir_result)
        self.assertTrue(any("final_state differs" in item for item in mismatches))

    def test_port_name_validation_is_exact(self) -> None:
        source = '''
def bad_ports(par_n):
    u8_i = 0
    u8_i = load(i=u8_i)
    return u8_i
'''
        harness = SimulationHarness(
            params={"par_n": 1},
            primitives={"load": PrimitiveModel(name="load", impl=lambda tb, addr: addr)},
            initial_state={},
        )

        with self.assertRaisesRegex(SimulationError, "port mismatch"):
            run_python(source, harness)

    def test_full_lu_core_matches_across_python_hir_lir(self) -> None:
        harness = SimulationHarness(
            params={"par_n": 3},
            primitives={
                "fetch_A": PrimitiveModel(name="fetch_A", impl=fetch_A, latency=2),
                "fetch_LU": PrimitiveModel(name="fetch_LU", impl=fetch_LU, latency=2),
                "store_LU": PrimitiveModel(name="store_LU", impl=store_LU, latency=1),
                "neg_comb": PrimitiveModel(name="neg_comb", impl=neg_comb),
                "fma": PrimitiveModel(name="fma", impl=fma, latency=3),
                "div": PrimitiveModel(name="div", impl=div, latency=4),
            },
            initial_state={
                "A": [
                    [4.0, 3.0, 2.0],
                    [6.0, 3.0, 0.0],
                    [2.0, 1.0, 1.0],
                ],
                "LU": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            },
        )
        report = run_all(LU_CORE, harness)

        self.assertTrue(report.ok, report.mismatches)
        self.assertEqual(report.python_result.final_state["LU"], report.hir_result.final_state["LU"])
        self.assertEqual(report.python_result.final_state["LU"], report.lir_result.final_state["LU"])
        self.assertEqual(report.python_result.locals, report.hir_result.locals)
        self.assertEqual(report.python_result.locals, report.lir_result.locals)
        neg_calls = [entry for entry in report.python_result.trace if entry.name == "neg_comb"]
        self.assertGreater(len(neg_calls), 0)
        self.assertTrue(all(entry.kind == "comb" for entry in neg_calls))
        self.assertGreater(report.hir_result.stats["latency_cycles"], 0)
        self.assertEqual(report.hir_result.stats["latency_cycles"], report.lir_result.stats["latency_cycles"])
        self.assertEqual(
            report.python_result.final_state,
            run_all(LU_CORE, harness).python_result.final_state,
        )


if __name__ == "__main__":
    unittest.main()