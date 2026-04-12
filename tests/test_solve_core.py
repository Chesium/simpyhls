import math
import unittest
from pathlib import Path

from compiler.primitive_rtl import PrimitiveRTLRegistry, PrimitiveRTLSpec
from compiler.sim_runtime import PrimitiveModel, SimulationHarness, run_all
from compiler.workflow import compile_file


ROOT = Path(__file__).resolve().parents[1]
SOLVE_CORE_PATH = ROOT / "examples" / "solve_core.dsl.py"
SOLVE_CORE = SOLVE_CORE_PATH.read_text()
SOLVE_CORE_DC_PATH = ROOT / "examples" / "solve_core_dc.dsl.py"
SOLVE_CORE_DC = SOLVE_CORE_DC_PATH.read_text()

GROUND_SENTINEL = 255
KIND_R = 1
KIND_I = 2
KIND_V = 3


def kind_code(kind_name: str) -> int:
    if kind_name == "R":
        return KIND_R
    if kind_name == "I":
        return KIND_I
    if kind_name == "V":
        return KIND_V
    raise ValueError(
        f"Unsupported solve_core element type '{kind_name}'. Supported types are 'R', 'I', and 'V'."
    )


def normalize_solver_netlist(netlist: list[tuple]) -> list[tuple]:
    sorted_netlist = sorted(netlist, key=lambda item: item[0])
    seen_indices: set[int] = set()
    normalized: list[tuple] = []

    for spec in sorted_netlist:
        if len(spec) != 5:
            raise ValueError(
                f"Expected raw solver tuple (index, type, node1, node2, value), got {spec!r}"
            )
        index, kind_name, node1, node2, value = spec
        if index in seen_indices:
            raise ValueError(f"Duplicate netlist index {index}")
        seen_indices.add(index)
        kind_code(kind_name)
        node0_idx = GROUND_SENTINEL if int(node1) == 0 else int(node1) - 1
        node1_idx = GROUND_SENTINEL if int(node2) == 0 else int(node2) - 1
        normalized.append((index, kind_name, node0_idx, node1_idx, float(value)))
    return normalized


def solver_dim(base_node_n: int, elements: list[tuple]) -> int:
    return base_node_n + sum(1 for _, kind_name, _, _, _ in elements if kind_name == "V")


def fetchElemKind(tb, idx):
    return kind_code(tb.state["elements"][idx][1])


def fetchElemN0(tb, idx):
    return tb.state["elements"][idx][2]


def fetchElemN1(tb, idx):
    return tb.state["elements"][idx][3]


def fetchElemVal0(tb, idx):
    return tb.state["elements"][idx][4]


def store_A(tb, i, j, v):
    tb.state["A"][i][j] = v


def fetch_A(tb, i, j):
    return tb.state["A"][i][j]


def accumA(tb, i, j, delta):
    tb.state["A"][i][j] += delta


def store_J(tb, i, v):
    tb.state["J"][i] = v


def fetch_J(tb, i):
    return tb.state["J"][i]


def accumJ(tb, i, delta):
    tb.state["J"][i] += delta


def fetch_LU(tb, i, j):
    return tb.state["LU"][i][j]


def store_LU(tb, i, j, v):
    tb.state["LU"][i][j] = v


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


def abs_comb(tb, v):
    return abs(v)


def gt_comb(tb, a, b):
    return a > b


def fma(tb, a, b, c):
    return (a * b) + c


def div(tb, a, b):
    return a / b


def solve_registry() -> PrimitiveRTLRegistry:
    return PrimitiveRTLRegistry(
        [
            PrimitiveRTLSpec(name="fetchElemKind", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemN0", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemN1", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemVal0", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="store_A", ports=("i", "j", "v"), latency=1),
            PrimitiveRTLSpec(name="fetch_A", ports=("i", "j"), result_port="result", latency=1),
            PrimitiveRTLSpec(name="accumA", ports=("i", "j", "delta"), latency=1),
            PrimitiveRTLSpec(name="store_J", ports=("i", "v"), latency=1),
            PrimitiveRTLSpec(name="fetch_J", ports=("i",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="accumJ", ports=("i", "delta"), latency=1),
            PrimitiveRTLSpec(name="fetch_LU", ports=("i", "j"), result_port="result", latency=1),
            PrimitiveRTLSpec(name="store_LU", ports=("i", "j", "v"), latency=1),
            PrimitiveRTLSpec(name="fetch_Y", ports=("i",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="store_Y", ports=("i", "v"), latency=1),
            PrimitiveRTLSpec(name="fetch_X", ports=("i",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="store_X", ports=("i", "v"), latency=1),
            PrimitiveRTLSpec(name="neg_comb", ports=("v",)),
            PrimitiveRTLSpec(name="abs_comb", ports=("v",)),
            PrimitiveRTLSpec(name="gt_comb", ports=("a", "b")),
            PrimitiveRTLSpec(name="fma", ports=("a", "b", "c"), result_port="result", latency=3),
            PrimitiveRTLSpec(name="div", ports=("a", "b"), result_port="result", latency=4),
        ]
    )


def solve_primitives() -> dict[str, PrimitiveModel]:
    return {
        "fetchElemKind": PrimitiveModel("fetchElemKind", fetchElemKind, latency=1),
        "fetchElemN0": PrimitiveModel("fetchElemN0", fetchElemN0, latency=1),
        "fetchElemN1": PrimitiveModel("fetchElemN1", fetchElemN1, latency=1),
        "fetchElemVal0": PrimitiveModel("fetchElemVal0", fetchElemVal0, latency=1),
        "store_A": PrimitiveModel("store_A", store_A, latency=1),
        "fetch_A": PrimitiveModel("fetch_A", fetch_A, latency=1),
        "accumA": PrimitiveModel("accumA", accumA, latency=1),
        "store_J": PrimitiveModel("store_J", store_J, latency=1),
        "fetch_J": PrimitiveModel("fetch_J", fetch_J, latency=1),
        "accumJ": PrimitiveModel("accumJ", accumJ, latency=1),
        "fetch_LU": PrimitiveModel("fetch_LU", fetch_LU, latency=1),
        "store_LU": PrimitiveModel("store_LU", store_LU, latency=1),
        "fetch_Y": PrimitiveModel("fetch_Y", fetch_Y, latency=1),
        "store_Y": PrimitiveModel("store_Y", store_Y, latency=1),
        "fetch_X": PrimitiveModel("fetch_X", fetch_X, latency=1),
        "store_X": PrimitiveModel("store_X", store_X, latency=1),
        "neg_comb": PrimitiveModel("neg_comb", neg_comb),
        "abs_comb": PrimitiveModel("abs_comb", abs_comb),
        "gt_comb": PrimitiveModel("gt_comb", gt_comb),
        "fma": PrimitiveModel("fma", fma, latency=3),
        "div": PrimitiveModel("div", div, latency=4),
    }


def make_state(dim: int, elements: list[tuple]) -> dict[str, object]:
    return {
        "elements": list(elements),
        "A": [[-123.0 for _ in range(dim)] for _ in range(dim)],
        "J": [-123.0 for _ in range(dim)],
        "LU": [[-123.0 for _ in range(dim)] for _ in range(dim)],
        "Y": [-123.0 for _ in range(dim)],
        "X": [-123.0 for _ in range(dim)],
    }


def make_harness(base_node_n: int, netlist: list[tuple]) -> SimulationHarness:
    elements = normalize_solver_netlist(netlist)
    dim = solver_dim(base_node_n, elements)
    return SimulationHarness(
        params={"par_elem_n": len(elements), "par_node_n": base_node_n},
        primitives=solve_primitives(),
        initial_state=make_state(dim, elements),
    )


def stamp_solver_reference(
    base_node_n: int, elements: list[tuple]
) -> tuple[list[list[float]], list[float], int]:
    dim = solver_dim(base_node_n, elements)
    A = [[0.0 for _ in range(dim)] for _ in range(dim)]
    J = [0.0 for _ in range(dim)]
    next_aux = base_node_n

    for _, kind_name, n0, n1, value in elements:

        if kind_name == "R":
            conductance = 1.0 / value
            if n0 != GROUND_SENTINEL:
                A[n0][n0] += conductance
                if n1 != GROUND_SENTINEL:
                    A[n0][n1] -= conductance
                    A[n1][n0] -= conductance
            if n1 != GROUND_SENTINEL:
                A[n1][n1] += conductance
            continue

        if kind_name == "I":
            if n0 != GROUND_SENTINEL:
                J[n0] -= value
            if n1 != GROUND_SENTINEL:
                J[n1] += value
            continue

        if kind_name == "V":
            aux = next_aux
            next_aux += 1
            if n0 != GROUND_SENTINEL:
                A[aux][n0] += 1.0
                A[n0][aux] -= 1.0
            if n1 != GROUND_SENTINEL:
                A[aux][n1] -= 1.0
                A[n1][aux] += 1.0
            J[aux] += value
            continue

        raise AssertionError(f"Unexpected element kind {kind_name}")

    return A, J, dim


def pack_lu_partial_pivot(
    A: list[list[float]], rhs: list[float]
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    n = len(A)
    A_perm = [list(map(float, row)) for row in A]
    rhs_perm = [float(value) for value in rhs]
    lu = [[0.0 for _ in range(n)] for _ in range(n)]

    for j in range(n):
        pivot = j
        best_abs = 0.0
        for row_idx in range(j, n):
            candidate = A_perm[row_idx][j]
            for k in range(j):
                candidate -= lu[row_idx][k] * lu[k][j]
            candidate_abs = abs(candidate)
            if row_idx == j or candidate_abs > best_abs:
                best_abs = candidate_abs
                pivot = row_idx

        if math.isclose(best_abs, 0.0, abs_tol=1e-12):
            raise ZeroDivisionError(f"Zero pivot at column {j}")
        if pivot != j:
            A_perm[j], A_perm[pivot] = A_perm[pivot], A_perm[j]
            rhs_perm[j], rhs_perm[pivot] = rhs_perm[pivot], rhs_perm[j]
            for k in range(j):
                lu[j][k], lu[pivot][k] = lu[pivot][k], lu[j][k]
        for i in range(n):
            acc = A_perm[i][j]
            for k in range(min(i, j)):
                acc -= lu[i][k] * lu[k][j]
            if i > j:
                acc /= lu[j][j]
            lu[i][j] = acc
    return A_perm, lu, rhs_perm


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


class SolveCoreTests(unittest.TestCase):
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

    def check_example(
        self,
        *,
        base_node_n: int,
        netlist: list[tuple],
        expected_solution: list[float],
        solution_places: int = 2,
        source: str = SOLVE_CORE,
    ) -> None:
        elements = normalize_solver_netlist(netlist)
        stamped_A, stamped_J, dim = stamp_solver_reference(base_node_n, elements)
        expected_A, expected_lu, expected_J = pack_lu_partial_pivot(stamped_A, stamped_J)
        expected_y = forward_substitute_packed(expected_lu, expected_J)
        expected_x = backward_substitute_packed(expected_lu, expected_y)

        self.assertEqual(dim, len(expected_solution))
        self.assertVectorClose(expected_x, expected_solution, places=solution_places)

        report = run_all(source, make_harness(base_node_n, netlist))
        self.assertTrue(report.ok, report.mismatches)

        for result in (report.python_result, report.hir_result, report.lir_result):
            self.assertMatrixClose(result.final_state["A"], expected_A)
            self.assertVectorClose(result.final_state["J"], expected_J)
            self.assertMatrixClose(result.final_state["LU"], expected_lu)
            self.assertVectorClose(result.final_state["Y"], expected_y)
            self.assertVectorClose(result.final_state["X"], expected_x)

        self.assertEqual(report.hir_result.stats["latency_cycles"], report.lir_result.stats["latency_cycles"])
        self.assertEqual(report.python_result.final_state["X"], report.hir_result.final_state["X"])
        self.assertEqual(report.python_result.final_state["X"], report.lir_result.final_state["X"])

    def test_normalize_solver_netlist_sorts_by_index(self) -> None:
        elements = normalize_solver_netlist(
            [
                (5, "R", 2, 0, 4.0),
                (1, "V", 1, 0, 5.0),
                (3, "I", 2, 3, 0.25),
            ]
        )

        self.assertEqual([item[0] for item in elements], [1, 3, 5])
        self.assertEqual(solver_dim(3, elements), 4)

    def test_normalize_solver_netlist_rejects_unsupported_element_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported solve_core element type"):
            normalize_solver_netlist([(0, "Z", 1, 2, 4.0)])

    def test_solve_core_matches_circuitsim_test1_voltage_divider(self) -> None:
        self.check_example(
            base_node_n=2,
            netlist=[
                (0, "V", 1, 0, 5.0),
                (1, "R", 1, 2, 3.0),
                (2, "R", 2, 0, 2.0),
            ],
            expected_solution=[5.0, 2.0, 1.0],
        )

    def test_solve_core_matches_circuitsim_test2_resistor_voltage_source_network(self) -> None:
        self.check_example(
            base_node_n=4,
            netlist=[
                (0, "V", 1, 0, 1.0),
                (1, "R", 1, 2, 50.0),
                (2, "R", 2, 3, 50.0),
                (3, "R", 3, 4, 50.0),
                (4, "R", 3, 0, 27.5),
                (5, "R", 2, 4, 90.92),
            ],
            expected_solution=[1.0, 0.5630, 0.2404, 0.3548, 0.0087],
            solution_places=4,
        )

    def test_solve_core_matches_circuitsim_test3_current_source_network(self) -> None:
        self.check_example(
            base_node_n=3,
            netlist=[
                (0, "I", 1, 2, 0.02),
                (1, "I", 2, 3, 0.06),
                (2, "R", 1, 2, 1300.0),
                (3, "R", 2, 3, 1000.0),
                (4, "R", 3, 0, 2000.0),
                (5, "R", 2, 0, 1500.0),
            ],
            expected_solution=[-46.0, -20.0, 26.67],
            solution_places=2,
        )

    def test_solve_core_dc_variant_matches_reference(self) -> None:
        self.check_example(
            base_node_n=2,
            netlist=[
                (0, "V", 1, 0, 5.0),
                (1, "R", 1, 2, 3.0),
                (2, "R", 2, 0, 2.0),
            ],
            expected_solution=[5.0, 2.0, 1.0],
            source=SOLVE_CORE_DC,
        )

    def test_solve_core_compiles_and_exposes_raw_riv_solver_interfaces(self) -> None:
        artifact = compile_file(SOLVE_CORE_PATH, solve_registry())

        self.assertEqual(artifact.report.function_name, "solve_core")
        self.assertIn("module solve_core", artifact.verilog)
        self.assertIn("input logic [31:0] par_elem_n", artifact.verilog)
        self.assertIn("input logic [31:0] par_node_n", artifact.verilog)
        self.assertIn("fetchElemKind_start", artifact.verilog)
        self.assertIn("fetchElemN0_start", artifact.verilog)
        self.assertIn("fetchElemN1_start", artifact.verilog)
        self.assertIn("fetchElemVal0_start", artifact.verilog)
        self.assertIn("store_A_start", artifact.verilog)
        self.assertIn("fetch_A_start", artifact.verilog)
        self.assertIn("store_J_start", artifact.verilog)
        self.assertIn("fetch_J_start", artifact.verilog)
        self.assertIn("fetch_LU_start", artifact.verilog)
        self.assertIn("store_LU_start", artifact.verilog)
        self.assertIn("store_Y_start", artifact.verilog)
        self.assertIn("fetch_X_start", artifact.verilog)
        self.assertIn("abs_comb(", artifact.verilog)
        self.assertIn("gt_comb(", artifact.verilog)
        self.assertNotIn("fetchElemAux_start", artifact.verilog)
        self.assertNotIn("fetchElemN2_start", artifact.verilog)
        self.assertNotIn("fetchElemN3_start", artifact.verilog)
        self.assertNotIn("fetchElemVal1_start", artifact.verilog)
        self.assertNotIn("fetchElemVal2_start", artifact.verilog)

    def test_solve_core_dc_variant_compiles_and_exposes_raw_riv_solver_interfaces(self) -> None:
        artifact = compile_file(SOLVE_CORE_DC_PATH, solve_registry())

        self.assertEqual(artifact.report.function_name, "solve_core_dc")
        self.assertIn("module solve_core_dc", artifact.verilog)
        self.assertIn("input logic [31:0] par_elem_n", artifact.verilog)
        self.assertIn("input logic [31:0] par_node_n", artifact.verilog)
        self.assertIn("fetchElemKind_start", artifact.verilog)
        self.assertIn("fetchElemN0_start", artifact.verilog)
        self.assertIn("fetchElemN1_start", artifact.verilog)
        self.assertIn("fetchElemVal0_start", artifact.verilog)
        self.assertIn("store_A_start", artifact.verilog)
        self.assertIn("fetch_A_start", artifact.verilog)
        self.assertIn("store_J_start", artifact.verilog)
        self.assertIn("fetch_J_start", artifact.verilog)
        self.assertIn("fetch_LU_start", artifact.verilog)
        self.assertIn("store_LU_start", artifact.verilog)
        self.assertIn("store_Y_start", artifact.verilog)
        self.assertIn("fetch_X_start", artifact.verilog)
        self.assertIn("abs_comb(", artifact.verilog)
        self.assertIn("gt_comb(", artifact.verilog)
        self.assertNotIn("fetchElemVal1_start", artifact.verilog)
        self.assertNotIn("fetchElemVal2_start", artifact.verilog)
        self.assertNotIn("fetchElemVal3_start", artifact.verilog)
        self.assertNotIn("fetch_prevX_start", artifact.verilog)


if __name__ == "__main__":
    unittest.main()
