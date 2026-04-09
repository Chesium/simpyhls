import math
import unittest
from pathlib import Path

from compiler.sim_runtime import PrimitiveModel, SimulationHarness, run_all
from compiler.primitive_rtl import PrimitiveRTLRegistry, PrimitiveRTLSpec
from compiler.workflow import compile_file


ROOT = Path(__file__).resolve().parents[1]
STAMPING_CORE_PATH = ROOT / "examples" / "stamping_core.dsl.py"
STAMPING_CORE = STAMPING_CORE_PATH.read_text()

GROUND = 65535
KIND_R = 1
KIND_I = 2
KIND_V = 3
KIND_VCVS = 4
KIND_NPN = 5


def _node_to_idx(node: int) -> int:
    return GROUND if node == 0 else node - 1


def preprocess_netlist(base_node_n: int, elements: list[tuple]) -> tuple[list[dict[str, object]], int]:
    next_aux = base_node_n
    lowered: list[dict[str, object]] = []

    for spec in elements:
        kind = spec[0]
        if kind == "R":
            _, i, j, resistance = spec
            lowered.append(
                {
                    "kind": KIND_R,
                    "n0": _node_to_idx(i),
                    "n1": _node_to_idx(j),
                    "n2": GROUND,
                    "n3": GROUND,
                    "aux": GROUND,
                    "v0": 1.0 / resistance,
                    "v1": 0.0,
                    "v2": 1.0,
                }
            )
            continue

        if kind == "I":
            _, i, j, current = spec
            lowered.append(
                {
                    "kind": KIND_I,
                    "n0": _node_to_idx(i),
                    "n1": _node_to_idx(j),
                    "n2": GROUND,
                    "n3": GROUND,
                    "aux": GROUND,
                    "v0": current,
                    "v1": 0.0,
                    "v2": 1.0,
                }
            )
            continue

        if kind == "V":
            _, i, j, voltage = spec
            lowered.append(
                {
                    "kind": KIND_V,
                    "n0": _node_to_idx(i),
                    "n1": _node_to_idx(j),
                    "n2": GROUND,
                    "n3": GROUND,
                    "aux": next_aux,
                    "v0": voltage,
                    "v1": 0.0,
                    "v2": 1.0,
                }
            )
            next_aux += 1
            continue

        if kind == "E":
            _, p, q, i, j, gain = spec
            lowered.append(
                {
                    "kind": KIND_VCVS,
                    "n0": _node_to_idx(p),
                    "n1": _node_to_idx(q),
                    "n2": _node_to_idx(i),
                    "n3": _node_to_idx(j),
                    "aux": next_aux,
                    "v0": gain,
                    "v1": 0.0,
                    "v2": 1.0,
                }
            )
            next_aux += 1
            continue

        if kind == "Q":
            _, b, c, e, beta, vbe_drop = spec
            lowered.append(
                {
                    "kind": KIND_NPN,
                    "n0": _node_to_idx(b),
                    "n1": _node_to_idx(c),
                    "n2": _node_to_idx(e),
                    "n3": GROUND,
                    "aux": next_aux,
                    "v0": beta,
                    "v1": vbe_drop,
                    "v2": 1.0,
                }
            )
            next_aux += 1
            continue

        raise ValueError(f"Unsupported element kind {kind!r}")

    return lowered, next_aux


def stamp_reference(elements: list[dict[str, object]], dim: int) -> tuple[list[list[float]], list[float]]:
    A = [[0.0 for _ in range(dim)] for _ in range(dim)]
    J = [0.0 for _ in range(dim)]

    for elem in elements:
        kind = elem["kind"]
        n0 = elem["n0"]
        n1 = elem["n1"]
        n2 = elem["n2"]
        n3 = elem["n3"]
        aux = elem["aux"]
        v0 = elem["v0"]
        v1 = elem["v1"]
        v2 = elem["v2"]

        if kind == KIND_R:
            if n0 != GROUND:
                A[n0][n0] += v0
                if n1 != GROUND:
                    A[n0][n1] -= v0
                    A[n1][n0] -= v0
            if n1 != GROUND:
                A[n1][n1] += v0
            continue

        if kind == KIND_I:
            if n0 != GROUND:
                J[n0] -= v0
            if n1 != GROUND:
                J[n1] += v0
            continue

        if kind == KIND_V:
            if n0 != GROUND:
                A[aux][n0] += v2
                A[n0][aux] -= v2
            if n1 != GROUND:
                A[aux][n1] -= v2
                A[n1][aux] += v2
            J[aux] += v0
            continue

        if kind == KIND_VCVS:
            if n2 != GROUND:
                A[aux][n2] += v2
                A[n2][aux] -= v2
            if n3 != GROUND:
                A[aux][n3] -= v2
                A[n3][aux] += v2
            if n0 != GROUND:
                A[aux][n0] -= v0
            if n1 != GROUND:
                A[aux][n1] += v0
            continue

        if kind == KIND_NPN:
            if n0 != GROUND:
                A[aux][n0] += v2
                A[n0][aux] -= v2
            if n2 != GROUND:
                A[aux][n2] -= v2
                A[n2][aux] += v2
                A[n2][aux] += v0
            if n1 != GROUND:
                A[n1][aux] -= v0
            J[aux] += v1
            continue

        raise ValueError(f"Unsupported lowered kind {kind!r}")

    return A, J


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


def fetchElemKind(tb, idx):
    return tb.state["elements"][idx]["kind"]


def fetchElemN0(tb, idx):
    return tb.state["elements"][idx]["n0"]


def fetchElemN1(tb, idx):
    return tb.state["elements"][idx]["n1"]


def fetchElemN2(tb, idx):
    return tb.state["elements"][idx]["n2"]


def fetchElemN3(tb, idx):
    return tb.state["elements"][idx]["n3"]


def fetchElemAux(tb, idx):
    return tb.state["elements"][idx]["aux"]


def fetchElemVal0(tb, idx):
    return tb.state["elements"][idx]["v0"]


def fetchElemVal1(tb, idx):
    return tb.state["elements"][idx]["v1"]


def fetchElemVal2(tb, idx):
    return tb.state["elements"][idx]["v2"]


def neg_comb(tb, v):
    return -v


def accumA(tb, i, j, delta):
    tb.state["A"][i][j] += delta


def accumJ(tb, i, delta):
    tb.state["J"][i] += delta


def stamping_registry() -> PrimitiveRTLRegistry:
    return PrimitiveRTLRegistry(
        [
            PrimitiveRTLSpec(name="fetchElemKind", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemN0", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemN1", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemN2", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemN3", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemAux", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemVal0", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemVal1", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemVal2", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="neg_comb", ports=("v",)),
            PrimitiveRTLSpec(name="accumA", ports=("i", "j", "delta"), latency=1),
            PrimitiveRTLSpec(name="accumJ", ports=("i", "delta"), latency=1),
        ]
    )


def make_harness(lowered: list[dict[str, object]], dim: int) -> SimulationHarness:
    return SimulationHarness(
        params={"par_elem_n": len(lowered)},
        primitives={
            "fetchElemKind": PrimitiveModel("fetchElemKind", fetchElemKind, latency=1),
            "fetchElemN0": PrimitiveModel("fetchElemN0", fetchElemN0, latency=1),
            "fetchElemN1": PrimitiveModel("fetchElemN1", fetchElemN1, latency=1),
            "fetchElemN2": PrimitiveModel("fetchElemN2", fetchElemN2, latency=1),
            "fetchElemN3": PrimitiveModel("fetchElemN3", fetchElemN3, latency=1),
            "fetchElemAux": PrimitiveModel("fetchElemAux", fetchElemAux, latency=1),
            "fetchElemVal0": PrimitiveModel("fetchElemVal0", fetchElemVal0, latency=1),
            "fetchElemVal1": PrimitiveModel("fetchElemVal1", fetchElemVal1, latency=1),
            "fetchElemVal2": PrimitiveModel("fetchElemVal2", fetchElemVal2, latency=1),
            "neg_comb": PrimitiveModel("neg_comb", neg_comb),
            "accumA": PrimitiveModel("accumA", accumA, latency=1),
            "accumJ": PrimitiveModel("accumJ", accumJ, latency=1),
        },
        initial_state={
            "elements": lowered,
            "A": [[0.0 for _ in range(dim)] for _ in range(dim)],
            "J": [0.0 for _ in range(dim)],
        },
    )


class StampingCoreTests(unittest.TestCase):
    def assertMatrixClose(self, actual: list[list[float]], expected: list[list[float]], places: int = 9) -> None:
        self.assertEqual(len(actual), len(expected))
        for row_idx, (actual_row, expected_row) in enumerate(zip(actual, expected)):
            self.assertEqual(len(actual_row), len(expected_row))
            for col_idx, (actual_value, expected_value) in enumerate(zip(actual_row, expected_row)):
                self.assertAlmostEqual(
                    actual_value,
                    expected_value,
                    places=places,
                    msg=f"Mismatch at A[{row_idx}][{col_idx}]",
                )

    def assertVectorClose(self, actual: list[float], expected: list[float], places: int = 9) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(
                actual_value,
                expected_value,
                places=places,
                msg=f"Mismatch at J[{idx}]",
            )

    def check_circuit(
        self,
        *,
        base_node_n: int,
        netlist: list[tuple],
        expected_solution: list[float] | None = None,
    ) -> None:
        lowered, dim = preprocess_netlist(base_node_n, netlist)
        harness = make_harness(lowered, dim)
        expected_A, expected_J = stamp_reference(lowered, dim)

        report = run_all(STAMPING_CORE, harness)
        self.assertTrue(report.ok, report.mismatches)

        for result in (report.python_result, report.hir_result, report.lir_result):
            self.assertMatrixClose(result.final_state["A"], expected_A)
            self.assertVectorClose(result.final_state["J"], expected_J)

        self.assertEqual(report.python_result.final_state, report.hir_result.final_state)
        self.assertEqual(report.python_result.final_state, report.lir_result.final_state)
        self.assertEqual(report.hir_result.stats["latency_cycles"], report.lir_result.stats["latency_cycles"])

        trace_names = [entry.name for entry in report.python_result.trace]
        self.assertIn("fetchElemKind", trace_names)
        self.assertIn("accumA", trace_names)

        if any(spec[0] == "I" for spec in netlist):
            self.assertIn("accumJ", trace_names)

        if any(spec[0] in {"R", "I", "E", "Q"} for spec in netlist):
            self.assertIn("neg_comb", trace_names)

        if expected_solution is not None:
            solved = solve_linear(expected_A, expected_J)
            self.assertEqual(len(solved), len(expected_solution))
            for idx, (actual_value, expected_value) in enumerate(zip(solved, expected_solution)):
                self.assertAlmostEqual(
                    actual_value,
                    expected_value,
                    places=2,
                    msg=f"Mismatch at solution[{idx}]",
                )

    def test_voltage_divider_matches_reference_stamp_and_solution(self) -> None:
        self.check_circuit(
            base_node_n=2,
            netlist=[
                ("V", 1, 0, 5.0),
                ("R", 1, 2, 3.0),
                ("R", 2, 0, 2.0),
            ],
            expected_solution=[5.0, 2.0, 1.0],
        )

    def test_current_source_circuit_matches_reference_stamp_and_solution(self) -> None:
        self.check_circuit(
            base_node_n=3,
            netlist=[
                ("I", 1, 2, 0.02),
                ("I", 2, 3, 0.06),
                ("R", 1, 2, 1300.0),
                ("R", 2, 3, 1000.0),
                ("R", 3, 0, 2000.0),
                ("R", 2, 0, 1500.0),
            ],
            expected_solution=[-46.0, -20.0, 26.6666666667],
        )

    def test_vcvs_circuit_matches_reference_stamp_and_solution(self) -> None:
        self.check_circuit(
            base_node_n=5,
            netlist=[
                ("V", 1, 0, 5.0),
                ("V", 2, 0, 3.0),
                ("R", 1, 3, 1.0),
                ("R", 2, 4, 2.0),
                ("R", 5, 0, 8.0),
                ("E", 3, 4, 5, 0, 4.0),
            ],
            expected_solution=[5.0, 3.0, 5.0, 3.0, 8.0, 0.0, 0.0, 1.0],
        )

    def test_npn_circuit_matches_reference_stamp_and_solution(self) -> None:
        self.check_circuit(
            base_node_n=4,
            netlist=[
                ("V", 1, 0, 3.7),
                ("V", 4, 0, 10.0),
                ("Q", 2, 3, 0, 100.0, 0.7),
                ("R", 1, 2, 10000.0),
                ("R", 3, 4, 220.0),
            ],
            expected_solution=[3.7, 0.7, 3.4, 10.0, 0.0003, 0.03, -0.0003],
        )

    def test_stamping_core_compiles_to_verilog_with_explicit_netlist_and_matrix_interfaces(self) -> None:
        artifact = compile_file(STAMPING_CORE_PATH, stamping_registry())

        self.assertEqual(artifact.report.function_name, "stamping_core")
        self.assertEqual(artifact.report.control_flow["for"], 1)
        self.assertEqual(artifact.report.control_flow["while"], 0)
        self.assertIn("module stamping_core", artifact.verilog)
        self.assertIn("fetchElemKind_start", artifact.verilog)
        self.assertIn("fetchElemVal0_start", artifact.verilog)
        self.assertIn("fetchElemVal2_start", artifact.verilog)
        self.assertIn("accumA_start", artifact.verilog)
        self.assertIn("accumJ_start", artifact.verilog)
        self.assertIn("neg_comb(", artifact.verilog)

        primitive_names = {item.name for item in artifact.report.primitive_usage}
        self.assertIn("fetchElemKind", primitive_names)
        self.assertIn("fetchElemVal0", primitive_names)
        self.assertIn("fetchElemVal2", primitive_names)
        self.assertIn("accumA", primitive_names)
        self.assertIn("accumJ", primitive_names)
        self.assertIn("neg_comb", primitive_names)


if __name__ == "__main__":
    unittest.main()
