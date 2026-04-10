import copy
import math
import unittest
from pathlib import Path

from compiler.hir_to_lir import lower_func
from compiler.ast_to_hir import lower_source
from compiler.primitive_rtl import PrimitiveRTLRegistry, PrimitiveRTLSpec
from compiler.sim_runtime import PrimitiveModel, SimulationHarness, run_all, run_hir, run_lir, run_python
from compiler.workflow import compile_file


ROOT = Path(__file__).resolve().parents[1]
LU_CORE_PATH = ROOT / "examples" / "lu_core.dsl.py"
FORWARD_CORE_PATH = ROOT / "examples" / "forward_sub_core.dsl.py"
BACKWARD_CORE_PATH = ROOT / "examples" / "backward_sub_core.dsl.py"

LU_CORE = LU_CORE_PATH.read_text()
FORWARD_CORE = FORWARD_CORE_PATH.read_text()
BACKWARD_CORE = BACKWARD_CORE_PATH.read_text()


def fetch_A(tb, i, j):
    return tb.state["A"][i][j]


def fetch_LU(tb, i, j):
    return tb.state["LU"][i][j]


def store_LU(tb, i, j, v):
    tb.state["LU"][i][j] = v


def fetch_J(tb, i):
    return tb.state["J"][i]


def fetch_Y(tb, i):
    return tb.state["Y"][i]


def store_Y(tb, i, v):
    tb.state["Y"][i] = v


def fetch_X(tb, i):
    return tb.state["X"][i]


def store_X(tb, i, v):
    tb.state["X"][i] = v


def neg_comb(tb, v):
    return -v


def fma(tb, a, b, c):
    return (a * b) + c


def div(tb, a, b):
    return a / b


def solver_registry() -> PrimitiveRTLRegistry:
    return PrimitiveRTLRegistry(
        [
            PrimitiveRTLSpec(name="fetch_A", ports=("i", "j"), result_port="result", latency=2),
            PrimitiveRTLSpec(name="fetch_LU", ports=("i", "j"), result_port="result", latency=2),
            PrimitiveRTLSpec(name="store_LU", ports=("i", "j", "v"), latency=1),
            PrimitiveRTLSpec(name="fetch_J", ports=("i",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetch_Y", ports=("i",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="store_Y", ports=("i", "v"), latency=1),
            PrimitiveRTLSpec(name="fetch_X", ports=("i",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="store_X", ports=("i", "v"), latency=1),
            PrimitiveRTLSpec(name="neg_comb", ports=("v",)),
            PrimitiveRTLSpec(name="fma", ports=("a", "b", "c"), result_port="result", latency=3),
            PrimitiveRTLSpec(name="div", ports=("a", "b"), result_port="result", latency=4),
        ]
    )


def solver_primitives() -> dict[str, PrimitiveModel]:
    return {
        "fetch_A": PrimitiveModel(name="fetch_A", impl=fetch_A, latency=2),
        "fetch_LU": PrimitiveModel(name="fetch_LU", impl=fetch_LU, latency=2),
        "store_LU": PrimitiveModel(name="store_LU", impl=store_LU, latency=1),
        "fetch_J": PrimitiveModel(name="fetch_J", impl=fetch_J, latency=1),
        "fetch_Y": PrimitiveModel(name="fetch_Y", impl=fetch_Y, latency=1),
        "store_Y": PrimitiveModel(name="store_Y", impl=store_Y, latency=1),
        "fetch_X": PrimitiveModel(name="fetch_X", impl=fetch_X, latency=1),
        "store_X": PrimitiveModel(name="store_X", impl=store_X, latency=1),
        "neg_comb": PrimitiveModel(name="neg_comb", impl=neg_comb),
        "fma": PrimitiveModel(name="fma", impl=fma, latency=3),
        "div": PrimitiveModel(name="div", impl=div, latency=4),
    }


def make_state(A: list[list[float]], J: list[float]) -> dict[str, object]:
    n = len(A)
    return {
        "A": [list(row) for row in A],
        "J": list(J),
        "LU": [[0.0 for _ in range(n)] for _ in range(n)],
        "Y": [0.0 for _ in range(n)],
        "X": [0.0 for _ in range(n)],
    }


def make_harness(par_n: int, state: dict[str, object]) -> SimulationHarness:
    return SimulationHarness(
        params={"par_n": par_n},
        primitives=solver_primitives(),
        initial_state=state,
    )


def pack_lu_no_pivot(A: list[list[float]]) -> list[list[float]]:
    n = len(A)
    lu = [[0.0 for _ in range(n)] for _ in range(n)]
    for j in range(n):
        for i in range(n):
            acc = A[i][j]
            for k in range(min(i, j)):
                acc -= lu[i][k] * lu[k][j]
            if i > j:
                acc /= lu[j][j]
            lu[i][j] = acc
    return lu


def forward_substitute_packed(lu: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(lu)
    y = [0.0 for _ in range(n)]
    for i in range(n):
        acc = rhs[i]
        for k in range(i):
            acc -= lu[i][k] * y[k]
        y[i] = acc
    return y


def backward_substitute_packed(lu: list[list[float]], y: list[float]) -> list[float]:
    n = len(lu)
    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        acc = y[i]
        for k in range(i + 1, n):
            acc -= lu[i][k] * x[k]
        x[i] = acc / lu[i][i]
    return x


def solve_linear(A: list[list[float]], J: list[float]) -> list[float]:
    n = len(A)
    aug = [list(map(float, row)) + [float(J[idx])] for idx, row in enumerate(A)]

    for pivot in range(n):
        best_row = max(range(pivot, n), key=lambda row_idx: abs(aug[row_idx][pivot]))
        if math.isclose(aug[best_row][pivot], 0.0, abs_tol=1e-12):
            raise ValueError("Singular matrix in reference solve")
        if best_row != pivot:
            aug[pivot], aug[best_row] = aug[best_row], aug[pivot]

        pivot_value = aug[pivot][pivot]
        for col in range(pivot, n + 1):
            aug[pivot][col] /= pivot_value

        for row_idx in range(n):
            if row_idx == pivot:
                continue
            factor = aug[row_idx][pivot]
            if math.isclose(factor, 0.0, abs_tol=1e-12):
                continue
            for col in range(pivot, n + 1):
                aug[row_idx][col] -= factor * aug[pivot][col]

    return [aug[row_idx][n] for row_idx in range(n)]


def run_stage(mode: str, source: str, harness: SimulationHarness):
    if mode == "python":
        return run_python(source, harness)
    hir = lower_source(source)
    if mode == "hir":
        return run_hir(hir, harness)
    if mode == "lir":
        return run_lir(lower_func(hir), harness)
    raise ValueError(f"Unsupported mode {mode!r}")


def run_solver_pipeline(mode: str, A: list[list[float]], J: list[float]) -> tuple[object, object, object]:
    par_n = len(A)
    lu_result = run_stage(mode, LU_CORE, make_harness(par_n, make_state(A, J)))
    forward_result = run_stage(mode, FORWARD_CORE, make_harness(par_n, lu_result.final_state))
    backward_result = run_stage(mode, BACKWARD_CORE, make_harness(par_n, forward_result.final_state))
    return lu_result, forward_result, backward_result


class LUSolverPipelineTests(unittest.TestCase):
    def assertVectorClose(self, actual: list[float], expected: list[float], places: int = 8) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(
                actual_value,
                expected_value,
                places=places,
                msg=f"Mismatch at vector index {idx}",
            )

    def assertMatrixClose(self, actual: list[list[float]], expected: list[list[float]], places: int = 8) -> None:
        self.assertEqual(len(actual), len(expected))
        for row_idx, (actual_row, expected_row) in enumerate(zip(actual, expected)):
            self.assertEqual(len(actual_row), len(expected_row))
            for col_idx, (actual_value, expected_value) in enumerate(zip(actual_row, expected_row)):
                self.assertAlmostEqual(
                    actual_value,
                    expected_value,
                    places=places,
                    msg=f"Mismatch at matrix entry ({row_idx}, {col_idx})",
                )

    def test_forward_sub_core_matches_reference_and_compiles(self) -> None:
        A = [
            [6.0, 1.0, 1.0],
            [1.0, 7.0, 1.0],
            [1.0, 1.0, 8.0],
        ]
        rhs = [9.0, 8.0, 7.0]
        lu = pack_lu_no_pivot(A)
        expected_y = forward_substitute_packed(lu, rhs)

        harness = make_harness(
            len(A),
            {
                "A": copy.deepcopy(A),
                "J": list(rhs),
                "LU": copy.deepcopy(lu),
                "Y": [0.0 for _ in range(len(A))],
                "X": [0.0 for _ in range(len(A))],
            },
        )
        report = run_all(FORWARD_CORE, harness)

        self.assertTrue(report.ok, report.mismatches)
        self.assertVectorClose(report.python_result.final_state["Y"], expected_y)
        self.assertVectorClose(report.hir_result.final_state["Y"], expected_y)
        self.assertVectorClose(report.lir_result.final_state["Y"], expected_y)
        self.assertEqual(report.hir_result.stats["latency_cycles"], report.lir_result.stats["latency_cycles"])

        artifact = compile_file(FORWARD_CORE_PATH, solver_registry())
        self.assertEqual(artifact.report.control_flow, {"if": 0, "for": 2, "while": 0})
        self.assertIn("module forward_sub_core", artifact.verilog)
        self.assertIn("fetch_J_start", artifact.verilog)
        self.assertIn("fetch_Y_start", artifact.verilog)
        self.assertIn("store_Y_start", artifact.verilog)

    def test_backward_sub_core_matches_reference_and_compiles(self) -> None:
        A = [
            [6.0, 1.0, 1.0],
            [1.0, 7.0, 1.0],
            [1.0, 1.0, 8.0],
        ]
        rhs = [9.0, 8.0, 7.0]
        lu = pack_lu_no_pivot(A)
        y = forward_substitute_packed(lu, rhs)
        expected_x = backward_substitute_packed(lu, y)

        harness = make_harness(
            len(A),
            {
                "A": copy.deepcopy(A),
                "J": list(rhs),
                "LU": copy.deepcopy(lu),
                "Y": list(y),
                "X": [0.0 for _ in range(len(A))],
            },
        )
        report = run_all(BACKWARD_CORE, harness)

        self.assertTrue(report.ok, report.mismatches)
        self.assertVectorClose(report.python_result.final_state["X"], expected_x)
        self.assertVectorClose(report.hir_result.final_state["X"], expected_x)
        self.assertVectorClose(report.lir_result.final_state["X"], expected_x)
        self.assertGreater(report.hir_result.stats["latency_cycles"], 0)
        self.assertEqual(report.hir_result.stats["latency_cycles"], report.lir_result.stats["latency_cycles"])

        artifact = compile_file(BACKWARD_CORE_PATH, solver_registry())
        self.assertEqual(artifact.report.control_flow, {"if": 0, "for": 0, "while": 2})
        self.assertIn("module backward_sub_core", artifact.verilog)
        self.assertIn("fetch_Y_start", artifact.verilog)
        self.assertIn("fetch_X_start", artifact.verilog)
        self.assertIn("store_X_start", artifact.verilog)

    def test_full_lu_forward_backward_pipeline_matches_reference_solution(self) -> None:
        A = [
            [9.0, 1.0, 2.0, 0.5],
            [1.0, 8.0, 1.0, 1.0],
            [2.0, 1.0, 7.0, 1.5],
            [0.5, 1.0, 1.5, 6.0],
        ]
        rhs = [12.0, 11.0, 13.0, 10.0]

        expected_lu = pack_lu_no_pivot(A)
        expected_y = forward_substitute_packed(expected_lu, rhs)
        expected_x = backward_substitute_packed(expected_lu, expected_y)
        direct_x = solve_linear(A, rhs)

        self.assertVectorClose(expected_x, direct_x, places=7)

        python_lu, python_forward, python_backward = run_solver_pipeline("python", A, rhs)
        hir_lu, hir_forward, hir_backward = run_solver_pipeline("hir", A, rhs)
        lir_lu, lir_forward, lir_backward = run_solver_pipeline("lir", A, rhs)

        self.assertMatrixClose(python_lu.final_state["LU"], expected_lu)
        self.assertMatrixClose(hir_lu.final_state["LU"], expected_lu)
        self.assertMatrixClose(lir_lu.final_state["LU"], expected_lu)

        self.assertVectorClose(python_forward.final_state["Y"], expected_y)
        self.assertVectorClose(hir_forward.final_state["Y"], expected_y)
        self.assertVectorClose(lir_forward.final_state["Y"], expected_y)

        self.assertVectorClose(python_backward.final_state["X"], expected_x)
        self.assertVectorClose(hir_backward.final_state["X"], expected_x)
        self.assertVectorClose(lir_backward.final_state["X"], expected_x)

        self.assertMatrixClose(python_lu.final_state["LU"], hir_lu.final_state["LU"])
        self.assertMatrixClose(python_lu.final_state["LU"], lir_lu.final_state["LU"])
        self.assertVectorClose(python_forward.final_state["Y"], hir_forward.final_state["Y"])
        self.assertVectorClose(python_forward.final_state["Y"], lir_forward.final_state["Y"])
        self.assertVectorClose(python_backward.final_state["X"], hir_backward.final_state["X"])
        self.assertVectorClose(python_backward.final_state["X"], lir_backward.final_state["X"])

        self.assertEqual(python_backward.final_state["A"], A)
        self.assertEqual(python_backward.final_state["J"], rhs)
        self.assertGreater(hir_backward.stats["latency_cycles"], 0)
        self.assertEqual(hir_backward.stats["latency_cycles"], lir_backward.stats["latency_cycles"])


if __name__ == "__main__":
    unittest.main()
