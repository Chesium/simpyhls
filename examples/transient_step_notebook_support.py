from __future__ import annotations

import html
import math
from pathlib import Path

from compiler.primitive_rtl import PrimitiveRTLRegistry, PrimitiveRTLSpec
from compiler.sim_runtime import PrimitiveModel, SimulationHarness, run_all
from compiler.workflow import compile_file


ROOT = Path(__file__).resolve().parents[1]
TRANSIENT_CORE_PATH = ROOT / "examples" / "transient_step_core.dsl.py"
TRANSIENT_CORE_SOURCE = TRANSIENT_CORE_PATH.read_text()

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
        "A": [[0.0 for _ in range(dim)] for _ in range(dim)],
        "J": [0.0 for _ in range(dim)],
        "LU": [[0.0 for _ in range(dim)] for _ in range(dim)],
        "Y": [0.0 for _ in range(dim)],
        "X": [0.0 for _ in range(dim)],
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


def compile_transient_kernel():
    return compile_file(TRANSIENT_CORE_PATH, transient_registry())


def make_uniform_schedule(total_time: float, dt: float) -> list[tuple[float, float]]:
    steps = int(round(total_time / dt))
    return [(step * dt, dt) for step in range(steps + 1)]


def make_variable_dt_schedule(entries: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(float(time), float(dt)) for time, dt in entries]


def run_transient_sequence(
    base_node_n: int,
    netlist: list[tuple],
    schedule: list[tuple[float, float]],
) -> dict[str, object]:
    elements = normalize_transient_netlist(netlist)
    dim = transient_dim(base_node_n, elements)
    state = make_state(dim, elements)
    steps: list[dict[str, object]] = []

    for step_index, (time_value, dt_value) in enumerate(schedule):
        report = run_all(
            TRANSIENT_CORE_SOURCE,
            make_harness(
                base_node_n,
                netlist,
                dt_value,
                time_value,
                initial_state=state,
            ),
        )
        if not report.ok:
            mismatch_text = "\n".join(report.mismatches)
            raise AssertionError(f"python/hir/lir mismatch at step {step_index}:\n{mismatch_text}")

        py_state = report.python_result.final_state
        state = py_state
        steps.append(
            {
                "step": step_index,
                "time": float(time_value),
                "dt": float(dt_value),
                "latency_cycles": int(report.python_result.stats["latency_cycles"]),
                "x_python": list(report.python_result.final_state["X"]),
                "x_hir": list(report.hir_result.final_state["X"]),
                "x_lir": list(report.lir_result.final_state["X"]),
                "prevX": list(py_state["prevX"]),
            }
        )

    return {
        "dim": dim,
        "elements": elements,
        "steps": steps,
        "final_state": state,
    }


def build_series(history: dict[str, object], node_indices: list[int]) -> list[dict[str, object]]:
    palette = ["#0b7285", "#c92a2a", "#2b8a3e", "#d9480f", "#5f3dc4", "#495057"]
    times = [step["time"] for step in history["steps"]]
    series: list[dict[str, object]] = []
    for pos, index in enumerate(node_indices):
        values = [step["x_python"][index] for step in history["steps"]]
        series.append(
            {
                "index": index,
                "times": times,
                "values": values,
                "color": palette[pos % len(palette)],
            }
        )
    return series


def render_result_table(
    history: dict[str, object],
    node_labels: dict[int, str] | None = None,
    *,
    max_rows: int = 12,
) -> str:
    if node_labels is None:
        node_labels = {}

    lines = ["step | time | dt | latency | values", "--- | --- | --- | --- | ---"]
    for step in history["steps"][:max_rows]:
        values = []
        for index, value in enumerate(step["x_python"]):
            label = node_labels.get(index, f"x[{index}]")
            values.append(f"{label}={value:.6g}")
        lines.append(
            f"{step['step']} | {step['time']:.6g} | {step['dt']:.6g} | "
            f"{step['latency_cycles']} | " + ", ".join(values)
        )
    if len(history["steps"]) > max_rows:
        lines.append(f"... | ... | ... | ... | {len(history['steps']) - max_rows} more rows")
    return "\n".join(lines)


def render_time_domain_svg(
    history: dict[str, object],
    node_indices: list[int],
    node_labels: dict[int, str] | None = None,
    *,
    title: str = "Transient response",
    annotations: list[dict[str, object]] | None = None,
    width: int = 960,
    height: int = 460,
) -> str:
    if node_labels is None:
        node_labels = {}
    if annotations is None:
        annotations = []

    series = build_series(history, node_indices)
    if not series or not history["steps"]:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='600' height='120'></svg>"

    margin_left = 78
    margin_right = 180
    margin_top = 44
    margin_bottom = 56
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = [time for item in series for time in item["times"]]
    y_values = [value for item in series for value in item["values"]]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
        y_min = y_min - 1.0
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def map_x(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def map_y(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>"
        "text { font-family: Consolas, 'Liberation Mono', monospace; }"
        ".axis { stroke: #555; stroke-width: 1; }"
        ".grid { stroke: #ddd; stroke-width: 1; }"
        ".legend-box { fill: #fafafa; stroke: #bbb; }"
        ".annotation { stroke: #999; stroke-dasharray: 4 4; }"
        "</style>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />",
        f"<text x='{margin_left}' y='24' font-size='18' font-weight='bold'>{html.escape(title)}</text>",
    ]

    for tick in range(6):
        x_value = x_min + ((x_max - x_min) * tick / 5.0)
        x_pos = map_x(x_value)
        parts.append(f"<line class='grid' x1='{x_pos:.2f}' y1='{margin_top}' x2='{x_pos:.2f}' y2='{margin_top + plot_height}' />")
        parts.append(f"<text x='{x_pos:.2f}' y='{height - 22}' text-anchor='middle' font-size='12'>{x_value:.3g}</text>")

    for tick in range(6):
        y_value = y_min + ((y_max - y_min) * tick / 5.0)
        y_pos = map_y(y_value)
        parts.append(f"<line class='grid' x1='{margin_left}' y1='{y_pos:.2f}' x2='{margin_left + plot_width}' y2='{y_pos:.2f}' />")
        parts.append(f"<text x='{margin_left - 10}' y='{y_pos + 4:.2f}' text-anchor='end' font-size='12'>{y_value:.3g}</text>")

    parts.append(f"<line class='axis' x1='{margin_left}' y1='{margin_top + plot_height}' x2='{margin_left + plot_width}' y2='{margin_top + plot_height}' />")
    parts.append(f"<line class='axis' x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' />")
    parts.append(f"<text x='{margin_left + plot_width / 2:.2f}' y='{height - 8}' text-anchor='middle' font-size='13'>time</text>")
    parts.append(f"<text x='18' y='{margin_top + plot_height / 2:.2f}' transform='rotate(-90 18 {margin_top + plot_height / 2:.2f})' text-anchor='middle' font-size='13'>state value</text>")

    for annotation in annotations:
        time_value = float(annotation["time"])
        if time_value < x_min or time_value > x_max:
            continue
        x_pos = map_x(time_value)
        label = html.escape(str(annotation["label"]))
        parts.append(f"<line class='annotation' x1='{x_pos:.2f}' y1='{margin_top}' x2='{x_pos:.2f}' y2='{margin_top + plot_height}' />")
        parts.append(f"<text x='{x_pos + 4:.2f}' y='{margin_top + 16}' font-size='12' fill='#666'>{label}</text>")

    for item in series:
        points = " ".join(f"{map_x(t):.2f},{map_y(v):.2f}" for t, v in zip(item["times"], item["values"]))
        label = html.escape(node_labels.get(item["index"], f"x[{item['index']}]"))
        last_x = map_x(item["times"][-1])
        last_y = map_y(item["values"][-1])
        parts.append(
            f"<polyline fill='none' stroke='{item['color']}' stroke-width='2.5' points='{points}' />"
        )
        parts.append(f"<circle cx='{last_x:.2f}' cy='{last_y:.2f}' r='4' fill='{item['color']}' />")
        parts.append(
            f"<text x='{last_x + 8:.2f}' y='{last_y - 6:.2f}' font-size='12' fill='{item['color']}'>{label}: {item['values'][-1]:.4g}</text>"
        )

    legend_x = margin_left + plot_width + 24
    legend_y = margin_top + 18
    legend_height = 26 * len(series) + 34
    parts.append(f"<rect class='legend-box' x='{legend_x - 12}' y='{legend_y - 16}' width='150' height='{legend_height}' rx='8' ry='8' />")
    parts.append(f"<text x='{legend_x}' y='{legend_y}' font-size='13' font-weight='bold'>Signals</text>")
    for offset, item in enumerate(series):
        y_pos = legend_y + 22 + 24 * offset
        label = html.escape(node_labels.get(item["index"], f"x[{item['index']}]"))
        parts.append(f"<line x1='{legend_x}' y1='{y_pos}' x2='{legend_x + 18}' y2='{y_pos}' stroke='{item['color']}' stroke-width='3' />")
        parts.append(f"<text x='{legend_x + 26}' y='{y_pos + 4}' font-size='12'>{label}</text>")

    parts.append("</svg>")
    return "".join(parts)


RC_CHARGE_CASE = {
    "name": "RC charge",
    "base_node_n": 2,
    "netlist": [
        (0, "V", 1, 0, 1.0),
        (1, "R", 1, 2, 1.0),
        (2, "C", 2, 0, 1.0),
    ],
    "node_labels": {
        0: "v_in",
        1: "v_cap",
        2: "i(Vsrc)",
    },
}


RL_DRIVE_CASE = {
    "name": "RL drive",
    "base_node_n": 2,
    "netlist": [
        (0, "V", 1, 0, 1.0),
        (1, "R", 1, 2, 2.0),
        (2, "L", 2, 0, 0.5),
    ],
    "node_labels": {
        0: "v_in",
        1: "v_L",
        2: "i(Vsrc)",
        3: "i_L",
    },
}


VSIN_CASE = {
    "name": "Sinusoidal drive",
    "base_node_n": 1,
    "netlist": [
        (0, "VSIN", 1, 0, 0.0, 1.0, 2.0 * math.pi, 0.0),
        (1, "R", 1, 0, 2.0),
    ],
    "node_labels": {
        0: "v_out",
        1: "i(VSIN)",
    },
}


VPWM_CASE = {
    "name": "PWM voltage drive",
    "base_node_n": 1,
    "netlist": [
        (0, "VPWM", 1, 0, 0.0, 2.5, 1.0, 0.35),
        (1, "R", 1, 0, 5.0),
    ],
    "node_labels": {
        0: "v_out",
        1: "i(VPWM)",
    },
}


PWM_SWITCH_CASE = {
    "name": "PWM switch divider",
    "base_node_n": 2,
    "netlist": [
        (0, "V", 1, 0, 1.0),
        (1, "R", 1, 2, 1.0),
        (2, "R", 2, 0, 10.0),
        (3, "SWPWM", 2, 0, 0.1, 1000.0, 1.0, 0.5),
    ],
    "node_labels": {
        0: "v_in",
        1: "v_sw",
        2: "i(Vsrc)",
    },
}


BOOST_LITE_CASE = {
    "name": "Boost-like switched RL stage",
    "base_node_n": 2,
    "netlist": [
        (0, "V", 1, 0, 1.0),
        (1, "L", 1, 2, 0.25),
        (2, "R", 2, 0, 4.0),
        (3, "SWPWM", 2, 0, 0.1, 1000.0, 0.4, 0.5),
    ],
    "node_labels": {
        0: "v_in",
        1: "v_sw",
        2: "i(Vsrc)",
        3: "i_L",
    },
}


VARIABLE_DT_SCHEDULE = [
    (0.00, 0.10),
    (0.10, 0.10),
    (0.20, 0.05),
    (0.25, 0.05),
    (0.30, 0.20),
    (0.50, 0.20),
    (0.70, 0.10),
    (0.80, 0.10),
]


__all__ = [
    "RC_CHARGE_CASE",
    "BOOST_LITE_CASE",
    "PWM_SWITCH_CASE",
    "RL_DRIVE_CASE",
    "TRANSIENT_CORE_PATH",
    "TRANSIENT_CORE_SOURCE",
    "VARIABLE_DT_SCHEDULE",
    "VPWM_CASE",
    "VSIN_CASE",
    "compile_transient_kernel",
    "make_uniform_schedule",
    "make_variable_dt_schedule",
    "normalize_transient_netlist",
    "render_result_table",
    "render_time_domain_svg",
    "run_transient_sequence",
    "transient_registry",
]
