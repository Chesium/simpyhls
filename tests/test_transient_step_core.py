import math
import unittest
from pathlib import Path

from compiler.primitive_rtl import PrimitiveRTLRegistry, PrimitiveRTLSpec
from compiler.sim_runtime import PrimitiveModel, SimulationHarness, run_all
from compiler.workflow import compile_file


ROOT = Path(__file__).resolve().parents[1]
TRANSIENT_CORE_PATH = ROOT / "examples" / "transient_step_core.dsl.py"
TRANSIENT_CORE = TRANSIENT_CORE_PATH.read_text()
SOLVE_CORE_TRANSIENT_PATH = ROOT / "examples" / "solve_core_transient.dsl.py"
SOLVE_CORE_TRANSIENT = SOLVE_CORE_TRANSIENT_PATH.read_text()

GROUND_SENTINEL = 65535
KIND_R = 1
KIND_I = 2
KIND_V = 3
KIND_C = 4
KIND_L = 5
KIND_VSIN = 6
KIND_ISIN = 7
KIND_SWPWM = 8
KIND_VPWM = 9


def kind_code(kind_name: str) -> int:
    if kind_name == "R":
        return KIND_R
    if kind_name == "I":
        return KIND_I
    if kind_name == "V":
        return KIND_V
    if kind_name == "C":
        return KIND_C
    if kind_name == "L":
        return KIND_L
    if kind_name == "VSIN":
        return KIND_VSIN
    if kind_name == "ISIN":
        return KIND_ISIN
    if kind_name == "SWPWM":
        return KIND_SWPWM
    if kind_name == "VPWM":
        return KIND_VPWM
    raise ValueError(
        "Unsupported transient_step_core element type "
        f"'{kind_name}'. Supported types are 'R', 'I', 'V', 'C', 'L', 'VSIN', 'ISIN', 'SWPWM', and 'VPWM'."
    )


def normalize_transient_netlist(netlist: list[tuple]) -> list[tuple]:
    sorted_netlist = sorted(netlist, key=lambda item: item[0])
    seen_indices: set[int] = set()
    normalized: list[tuple] = []

    for spec in sorted_netlist:
        if len(spec) == 5:
            index, kind_name, node1, node2, value = spec
            values = (float(value), 0.0, 0.0, 0.0)
        elif len(spec) == 8:
            index, kind_name, node1, node2, value0, value1, value2, value3 = spec
            values = (float(value0), float(value1), float(value2), float(value3))
        else:
            raise ValueError(
                "Expected transient tuple "
                "(index, type, node1, node2, value) or "
                "(index, type, node1, node2, value0, value1, value2, value3), "
                f"got {spec!r}"
            )
        if index in seen_indices:
            raise ValueError(f"Duplicate netlist index {index}")
        seen_indices.add(index)
        kind_code(kind_name)
        normalized.append((index, kind_name, int(node1), int(node2), *values))
    return normalized


def transient_dim(base_node_n: int, elements: list[tuple]) -> int:
    return base_node_n + sum(1 for _, kind_name, *_ in elements if kind_name in {"V", "L", "VSIN", "VPWM"})


def fetchElemKind(tb, idx):
    return kind_code(tb.state["elements"][idx][1])


def fetchElemN0(tb, idx):
    return tb.state["elements"][idx][2]


def fetchElemN1(tb, idx):
    return tb.state["elements"][idx][3]


def fetchElemVal0(tb, idx):
    return tb.state["elements"][idx][4]


def fetchElemVal1(tb, idx):
    return tb.state["elements"][idx][5]


def fetchElemVal2(tb, idx):
    return tb.state["elements"][idx][6]


def fetchElemVal3(tb, idx):
    return tb.state["elements"][idx][7]


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


def fetch_prevX(tb, i):
    return tb.state["prevX"][i]


def store_prevX(tb, i, v):
    tb.state["prevX"][i] = v


def neg_comb(tb, v):
    return -v


def abs_comb(tb, v):
    return abs(v)


def gt_comb(tb, a, b):
    return a > b


def sin_comb(tb, v):
    return math.sin(v)


def pwm_gate_comb(tb, time, period, duty):
    if period <= 0.0:
        raise ValueError("PWM period must be positive")
    if duty <= 0.0:
        return 0
    if duty >= 1.0:
        return 1
    phase = time % period
    return 1 if phase < (period * duty) else 0


def fma(tb, a, b, c):
    return (a * b) + c


def div(tb, a, b):
    return a / b


def transient_registry() -> PrimitiveRTLRegistry:
    return PrimitiveRTLRegistry(
        [
            PrimitiveRTLSpec(name="fetchElemKind", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemN0", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemN1", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemVal0", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemVal1", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemVal2", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchElemVal3", ports=("idx",), result_port="result", latency=1),
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
            PrimitiveRTLSpec(name="fetch_prevX", ports=("i",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="store_prevX", ports=("i", "v"), latency=1),
            PrimitiveRTLSpec(name="neg_comb", ports=("v",)),
            PrimitiveRTLSpec(name="abs_comb", ports=("v",)),
            PrimitiveRTLSpec(name="gt_comb", ports=("a", "b")),
            PrimitiveRTLSpec(name="sin_comb", ports=("v",)),
            PrimitiveRTLSpec(name="pwm_gate_comb", ports=("time", "period", "duty")),
            PrimitiveRTLSpec(name="fma", ports=("a", "b", "c"), result_port="result", latency=3),
            PrimitiveRTLSpec(name="div", ports=("a", "b"), result_port="result", latency=4),
        ]
    )


def transient_primitives() -> dict[str, PrimitiveModel]:
    return {
        "fetchElemKind": PrimitiveModel("fetchElemKind", fetchElemKind, latency=1),
        "fetchElemN0": PrimitiveModel("fetchElemN0", fetchElemN0, latency=1),
        "fetchElemN1": PrimitiveModel("fetchElemN1", fetchElemN1, latency=1),
        "fetchElemVal0": PrimitiveModel("fetchElemVal0", fetchElemVal0, latency=1),
        "fetchElemVal1": PrimitiveModel("fetchElemVal1", fetchElemVal1, latency=1),
        "fetchElemVal2": PrimitiveModel("fetchElemVal2", fetchElemVal2, latency=1),
        "fetchElemVal3": PrimitiveModel("fetchElemVal3", fetchElemVal3, latency=1),
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
        "fetch_prevX": PrimitiveModel("fetch_prevX", fetch_prevX, latency=1),
        "store_prevX": PrimitiveModel("store_prevX", store_prevX, latency=1),
        "neg_comb": PrimitiveModel("neg_comb", neg_comb),
        "abs_comb": PrimitiveModel("abs_comb", abs_comb),
        "gt_comb": PrimitiveModel("gt_comb", gt_comb),
        "sin_comb": PrimitiveModel("sin_comb", sin_comb),
        "pwm_gate_comb": PrimitiveModel("pwm_gate_comb", pwm_gate_comb),
        "fma": PrimitiveModel("fma", fma, latency=3),
        "div": PrimitiveModel("div", div, latency=4),
    }


def make_state(dim: int, elements: list[tuple], prev_x: list[float] | None = None) -> dict[str, object]:
    if prev_x is None:
        prev_x = [0.0 for _ in range(dim)]
    return {
        "elements": list(elements),
        "A": [[-321.0 for _ in range(dim)] for _ in range(dim)],
        "J": [-321.0 for _ in range(dim)],
        "LU": [[-321.0 for _ in range(dim)] for _ in range(dim)],
        "Y": [-321.0 for _ in range(dim)],
        "X": [-321.0 for _ in range(dim)],
        "prevX": list(prev_x),
    }


def make_harness(
    base_node_n: int,
    netlist: list[tuple],
    dt: float,
    time: float,
    *,
    initial_state: dict[str, object] | None = None,
    prev_x: list[float] | None = None,
) -> SimulationHarness:
    elements = normalize_transient_netlist(netlist)
    dim = transient_dim(base_node_n, elements)
    state = initial_state if initial_state is not None else make_state(dim, elements, prev_x)
    return SimulationHarness(
        params={
            "par_elem_n": len(elements),
            "par_node_n": base_node_n,
            "par_dt": float(dt),
            "par_time": float(time),
        },
        primitives=transient_primitives(),
        initial_state=state,
    )


def node_to_idx(node: int) -> int:
    return GROUND_SENTINEL if node == 0 else node - 1


def source_value(kind_name: str, element: tuple, time: float) -> float:
    value0, value1, value2, value3 = element[4], element[5], element[6], element[7]
    if kind_name in {"V", "I"}:
        return value0
    if kind_name in {"VSIN", "ISIN"}:
        return value0 + (value1 * math.sin((value2 * time) + value3))
    if kind_name == "VPWM":
        return value1 if pwm_gate_comb(None, time, value2, value3) != 0 else value0
    raise AssertionError(f"Unexpected source kind {kind_name}")


def stamp_transient_reference(
    base_node_n: int,
    elements: list[tuple],
    dt: float,
    time: float,
    prev_x: list[float],
) -> tuple[list[list[float]], list[float], int]:
    dim = transient_dim(base_node_n, elements)
    A = [[0.0 for _ in range(dim)] for _ in range(dim)]
    J = [0.0 for _ in range(dim)]
    next_aux = base_node_n

    for element in elements:
        _, kind_name, node1, node2, value0, _, _, _ = element
        n0 = node_to_idx(node1)
        n1 = node_to_idx(node2)

        if kind_name == "R":
            conductance = 1.0 / value0
            if n0 != GROUND_SENTINEL:
                A[n0][n0] += conductance
                if n1 != GROUND_SENTINEL:
                    A[n0][n1] -= conductance
                    A[n1][n0] -= conductance
            if n1 != GROUND_SENTINEL:
                A[n1][n1] += conductance
            continue

        if kind_name == "I":
            value = source_value(kind_name, element, time)
            if n0 != GROUND_SENTINEL:
                J[n0] -= value
            if n1 != GROUND_SENTINEL:
                J[n1] += value
            continue

        if kind_name == "V":
            value = source_value(kind_name, element, time)
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

        if kind_name == "C":
            conductance = value0 / dt
            v_prev = 0.0
            if n0 != GROUND_SENTINEL:
                v_prev += prev_x[n0]
            if n1 != GROUND_SENTINEL:
                v_prev -= prev_x[n1]
            if n0 != GROUND_SENTINEL:
                A[n0][n0] += conductance
                if n1 != GROUND_SENTINEL:
                    A[n0][n1] -= conductance
                    A[n1][n0] -= conductance
            if n1 != GROUND_SENTINEL:
                A[n1][n1] += conductance
            history = conductance * v_prev
            if n0 != GROUND_SENTINEL:
                J[n0] += history
            if n1 != GROUND_SENTINEL:
                J[n1] -= history
            continue

        if kind_name == "L":
            aux = next_aux
            next_aux += 1
            branch_scale = value0 / dt
            prev_i = prev_x[aux]
            if n0 != GROUND_SENTINEL:
                A[aux][n0] += 1.0
                A[n0][aux] -= 1.0
            if n1 != GROUND_SENTINEL:
                A[aux][n1] -= 1.0
                A[n1][aux] += 1.0
            A[aux][aux] -= branch_scale
            J[aux] -= branch_scale * prev_i
            continue

        if kind_name == "VSIN":
            value = source_value(kind_name, element, time)
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

        if kind_name == "ISIN":
            value = source_value(kind_name, element, time)
            if n0 != GROUND_SENTINEL:
                J[n0] -= value
            if n1 != GROUND_SENTINEL:
                J[n1] += value
            continue

        if kind_name == "SWPWM":
            resistance = value0 if pwm_gate_comb(None, time, element[6], element[7]) != 0 else element[5]
            conductance = 1.0 / resistance
            if n0 != GROUND_SENTINEL:
                A[n0][n0] += conductance
                if n1 != GROUND_SENTINEL:
                    A[n0][n1] -= conductance
                    A[n1][n0] -= conductance
            if n1 != GROUND_SENTINEL:
                A[n1][n1] += conductance
            continue

        if kind_name == "VPWM":
            value = source_value(kind_name, element, time)
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


def solve_dense_with_partial_pivot(
    A: list[list[float]], rhs: list[float]
) -> tuple[list[list[float]], list[float], list[float]]:
    n = len(A)
    working = [list(map(float, row)) for row in A]
    rhs_working = [float(value) for value in rhs]
    swapped_A = [list(map(float, row)) for row in A]
    swapped_rhs = [float(value) for value in rhs]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(working[row][col]))
        if math.isclose(working[pivot][col], 0.0, abs_tol=1e-12):
            raise ZeroDivisionError(f"Zero pivot at column {col}")
        if pivot != col:
            working[col], working[pivot] = working[pivot], working[col]
            rhs_working[col], rhs_working[pivot] = rhs_working[pivot], rhs_working[col]
            swapped_A[col], swapped_A[pivot] = swapped_A[pivot], swapped_A[col]
            swapped_rhs[col], swapped_rhs[pivot] = swapped_rhs[pivot], swapped_rhs[col]

        for row in range(col + 1, n):
            factor = working[row][col] / working[col][col]
            working[row][col] = 0.0
            for inner in range(col + 1, n):
                working[row][inner] -= factor * working[col][inner]
            rhs_working[row] -= factor * rhs_working[col]

    x = [0.0 for _ in range(n)]
    for row in range(n - 1, -1, -1):
        acc = rhs_working[row]
        for col in range(row + 1, n):
            acc -= working[row][col] * x[col]
        x[row] = acc / working[row][row]

    return swapped_A, swapped_rhs, x


def solve_reference_step(
    base_node_n: int,
    netlist: list[tuple],
    dt: float,
    time: float,
    prev_x: list[float] | None = None,
) -> dict[str, object]:
    elements = normalize_transient_netlist(netlist)
    dim = transient_dim(base_node_n, elements)
    if prev_x is None:
        prev_x = [0.0 for _ in range(dim)]
    stamped_A, stamped_J, _ = stamp_transient_reference(base_node_n, elements, dt, time, prev_x)
    expected_A, expected_J, expected_x = solve_dense_with_partial_pivot(stamped_A, stamped_J)
    return {
        "A": expected_A,
        "J": expected_J,
        "X": expected_x,
        "prevX": list(expected_x),
        "dim": dim,
    }


class TransientStepCoreTests(unittest.TestCase):
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

    def assertStepMatchesReference(
        self,
        *,
        base_node_n: int,
        netlist: list[tuple],
        dt: float,
        time: float,
        prev_x: list[float] | None = None,
        initial_state: dict[str, object] | None = None,
        source: str = TRANSIENT_CORE,
    ):
        reference = solve_reference_step(base_node_n, netlist, dt, time, prev_x)
        report = run_all(
            source,
            make_harness(
                base_node_n,
                netlist,
                dt,
                time,
                initial_state=initial_state,
                prev_x=prev_x,
            ),
        )
        self.assertTrue(report.ok, report.mismatches)

        for result in (report.python_result, report.hir_result, report.lir_result):
            self.assertVectorClose(result.final_state["X"], reference["X"])
            self.assertVectorClose(result.final_state["prevX"], reference["prevX"])

        self.assertEqual(report.hir_result.stats["latency_cycles"], report.lir_result.stats["latency_cycles"])
        return report, reference

    def test_normalize_transient_netlist_supports_dc_and_sinusoidal_shapes(self) -> None:
        elements = normalize_transient_netlist(
            [
                (5, "R", 2, 0, 4.0),
                (1, "VSIN", 1, 0, 0.5, 1.5, 2.0, 0.25),
                (3, "L", 2, 3, 0.1),
            ]
        )

        self.assertEqual([item[0] for item in elements], [1, 3, 5])
        self.assertEqual(transient_dim(3, elements), 5)
        self.assertEqual(elements[0][4:], (0.5, 1.5, 2.0, 0.25))
        self.assertEqual(elements[1][4:], (0.1, 0.0, 0.0, 0.0))

    def test_normalize_transient_netlist_supports_pwm_shapes(self) -> None:
        elements = normalize_transient_netlist(
            [
                (4, "VPWM", 1, 0, 0.0, 3.3, 1.0, 0.25),
                (2, "SWPWM", 2, 0, 0.1, 1000.0, 1.0, 0.5),
            ]
        )

        self.assertEqual([item[0] for item in elements], [2, 4])
        self.assertEqual(elements[0][4:], (0.1, 1000.0, 1.0, 0.5))
        self.assertEqual(elements[1][4:], (0.0, 3.3, 1.0, 0.25))

    def test_transient_step_core_rejects_unsupported_element_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported transient_step_core element type"):
            normalize_transient_netlist([(0, "Q", 1, 2, 0.1)])

    def test_rc_single_step_matches_reference(self) -> None:
        report, reference = self.assertStepMatchesReference(
            base_node_n=2,
            netlist=[
                (0, "V", 1, 0, 1.0),
                (1, "R", 1, 2, 1.0),
                (2, "C", 2, 0, 1.0),
            ],
            dt=0.1,
            time=0.0,
        )

        self.assertAlmostEqual(reference["X"][1], 1.0 / 11.0, places=8)
        self.assertEqual(reference["prevX"], report.python_result.final_state["prevX"])

    def test_rl_single_step_matches_reference(self) -> None:
        self.assertStepMatchesReference(
            base_node_n=2,
            netlist=[
                (0, "V", 1, 0, 1.0),
                (1, "R", 1, 2, 2.0),
                (2, "L", 2, 0, 0.5),
            ],
            dt=0.1,
            time=0.0,
        )

    def test_sinusoidal_voltage_source_matches_reference(self) -> None:
        _, reference = self.assertStepMatchesReference(
            base_node_n=1,
            netlist=[
                (0, "VSIN", 1, 0, 0.5, 1.5, 2.0, 0.25),
                (1, "R", 1, 0, 2.0),
            ],
            dt=0.05,
            time=0.75,
        )

        expected_source = 0.5 + (1.5 * math.sin((2.0 * 0.75) + 0.25))
        self.assertAlmostEqual(reference["X"][0], expected_source, places=8)

    def test_sinusoidal_current_source_matches_reference(self) -> None:
        _, reference = self.assertStepMatchesReference(
            base_node_n=1,
            netlist=[
                (0, "ISIN", 1, 0, 0.25, 0.75, math.pi, 0.0),
                (1, "R", 1, 0, 4.0),
            ],
            dt=0.05,
            time=0.5,
        )

        expected_current = 0.25 + (0.75 * math.sin(math.pi * 0.5))
        self.assertAlmostEqual(reference["X"][0], -4.0 * expected_current, places=8)

    def test_pwm_voltage_source_toggles_between_low_and_high_levels(self) -> None:
        _, reference_high = self.assertStepMatchesReference(
            base_node_n=1,
            netlist=[
                (0, "VPWM", 1, 0, 0.0, 2.5, 1.0, 0.25),
                (1, "R", 1, 0, 5.0),
            ],
            dt=0.01,
            time=0.10,
        )
        _, reference_low = self.assertStepMatchesReference(
            base_node_n=1,
            netlist=[
                (0, "VPWM", 1, 0, 0.0, 2.5, 1.0, 0.25),
                (1, "R", 1, 0, 5.0),
            ],
            dt=0.01,
            time=0.60,
        )

        self.assertAlmostEqual(reference_high["X"][0], 2.5, places=8)
        self.assertAlmostEqual(reference_low["X"][0], 0.0, places=8)

    def test_pwm_switch_changes_node_voltage_between_on_and_off_phases(self) -> None:
        netlist = [
            (0, "V", 1, 0, 1.0),
            (1, "R", 1, 2, 1.0),
            (2, "R", 2, 0, 10.0),
            (3, "SWPWM", 2, 0, 0.1, 1000.0, 1.0, 0.5),
        ]
        _, reference_on = self.assertStepMatchesReference(
            base_node_n=2,
            netlist=netlist,
            dt=0.01,
            time=0.10,
        )
        _, reference_off = self.assertStepMatchesReference(
            base_node_n=2,
            netlist=netlist,
            dt=0.01,
            time=0.75,
        )

        self.assertLess(reference_on["X"][1], 0.15)
        self.assertGreater(reference_off["X"][1], 0.8)

    def test_pwm_switch_can_drive_an_rl_stage_over_multiple_steps(self) -> None:
        base_node_n = 2
        netlist = [
            (0, "V", 1, 0, 1.0),
            (1, "L", 1, 2, 0.25),
            (2, "R", 2, 0, 4.0),
            (3, "SWPWM", 2, 0, 0.1, 1000.0, 0.4, 0.5),
        ]
        elements = normalize_transient_netlist(netlist)
        dim = transient_dim(base_node_n, elements)
        state = make_state(dim, elements)
        prev_x = [0.0 for _ in range(dim)]

        sampled_currents: list[float] = []
        for time, dt in [(0.00, 0.02), (0.02, 0.02), (0.04, 0.02), (0.06, 0.02), (0.08, 0.02)]:
            report, reference = self.assertStepMatchesReference(
                base_node_n=base_node_n,
                netlist=netlist,
                dt=dt,
                time=time,
                prev_x=prev_x,
                initial_state=state,
            )
            state = report.python_result.final_state
            prev_x = list(reference["X"])
            sampled_currents.append(reference["X"][2])

        self.assertTrue(any(abs(current) > 1e-6 for current in sampled_currents))
        self.assertNotAlmostEqual(sampled_currents[1], sampled_currents[-1], places=6)

    def test_multi_step_variable_dt_matches_reference_and_updates_history(self) -> None:
        base_node_n = 2
        netlist = [
            (0, "V", 1, 0, 1.0),
            (1, "R", 1, 2, 1.0),
            (2, "C", 2, 0, 1.0),
        ]
        elements = normalize_transient_netlist(netlist)
        dim = transient_dim(base_node_n, elements)
        state = make_state(dim, elements)
        prev_x = [0.0 for _ in range(dim)]

        for time, dt in [(0.0, 0.1), (0.1, 0.1), (0.2, 0.2), (0.4, 0.05)]:
            report, reference = self.assertStepMatchesReference(
                base_node_n=base_node_n,
                netlist=netlist,
                dt=dt,
                time=time,
                prev_x=prev_x,
                initial_state=state,
            )
            state = report.python_result.final_state
            prev_x = list(reference["X"])
            self.assertVectorClose(state["prevX"], prev_x)

        self.assertGreater(prev_x[1], 0.0)
        self.assertLess(prev_x[1], 1.0)

    def test_solve_core_transient_variant_matches_reference(self) -> None:
        _, reference = self.assertStepMatchesReference(
            base_node_n=2,
            netlist=[
                (0, "V", 1, 0, 1.0),
                (1, "R", 1, 2, 1.0),
                (2, "C", 2, 0, 1.0),
            ],
            dt=0.1,
            time=0.0,
            source=SOLVE_CORE_TRANSIENT,
        )

        self.assertAlmostEqual(reference["X"][1], 1.0 / 11.0, places=8)

    def test_transient_step_core_compiles_and_exposes_history_and_sine_interfaces(self) -> None:
        artifact = compile_file(TRANSIENT_CORE_PATH, transient_registry())

        self.assertEqual(artifact.report.function_name, "transient_step_core")
        self.assertIn("module transient_step_core", artifact.verilog)
        self.assertIn("input logic [31:0] par_dt", artifact.verilog)
        self.assertIn("input logic [31:0] par_time", artifact.verilog)
        self.assertIn("fetchElemVal1_start", artifact.verilog)
        self.assertIn("fetchElemVal2_start", artifact.verilog)
        self.assertIn("fetchElemVal3_start", artifact.verilog)
        self.assertIn("fetch_prevX_start", artifact.verilog)
        self.assertIn("store_prevX_start", artifact.verilog)
        self.assertIn("sin_comb(", artifact.verilog)
        self.assertIn("pwm_gate_comb(", artifact.verilog)
        self.assertIn("abs_comb(", artifact.verilog)
        self.assertIn("gt_comb(", artifact.verilog)

    def test_solve_core_transient_variant_compiles_and_exposes_history_and_sine_interfaces(self) -> None:
        artifact = compile_file(SOLVE_CORE_TRANSIENT_PATH, transient_registry())

        self.assertEqual(artifact.report.function_name, "solve_core_transient")
        self.assertIn("module solve_core_transient", artifact.verilog)
        self.assertIn("input logic [31:0] par_dt", artifact.verilog)
        self.assertIn("input logic [31:0] par_time", artifact.verilog)
        self.assertIn("fetchElemVal1_start", artifact.verilog)
        self.assertIn("fetchElemVal2_start", artifact.verilog)
        self.assertIn("fetchElemVal3_start", artifact.verilog)
        self.assertIn("fetch_prevX_start", artifact.verilog)
        self.assertIn("store_prevX_start", artifact.verilog)
        self.assertIn("sin_comb(", artifact.verilog)
        self.assertIn("pwm_gate_comb(", artifact.verilog)


if __name__ == "__main__":
    unittest.main()
