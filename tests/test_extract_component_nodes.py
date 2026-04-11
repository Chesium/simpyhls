import unittest
from pathlib import Path

from compiler.primitive_rtl import PrimitiveRTLRegistry, PrimitiveRTLSpec
from compiler.sim_runtime import PrimitiveModel, SimulationHarness, run_all
from compiler.workflow import compile_file


ROOT = Path(__file__).resolve().parents[1]
EXTRACT_COMPONENT_NODES_PATH = ROOT / "examples" / "extract_component_nodes.dsl.py"
EXTRACT_COMPONENT_NODES = EXTRACT_COMPONENT_NODES_PATH.read_text()


TYPE_RL = 5
TYPE_RR = 6
TYPE_VL = 7
TYPE_VR = 8
TYPE_IL = 9
TYPE_IR = 10
TYPE_LL = 11
TYPE_LR = 12
TYPE_CL = 13
TYPE_CR = 14
TYPE_GROUND = 15


def is_two_terminal_type(type_code: int) -> bool:
    return TYPE_RL <= type_code <= TYPE_CR


def is_ground_type(type_code: int) -> bool:
    return type_code == TYPE_GROUND


def opp_dir(direction: int) -> int:
    if direction == 0:
        return 2
    if direction == 1:
        return 3
    if direction == 2:
        return 0
    return 1


def step(x: int, y: int, direction: int) -> tuple[int, int]:
    if direction == 0:
        return x + 1, y
    if direction == 1:
        return x, y + 1
    if direction == 2:
        return x - 1, y
    return x, y - 1


def region_at(grid: list[list[int]], x: int, y: int, width: int, height: int) -> int:
    if x < 0 or y < 0 or x >= width or y >= height:
        return 0
    return int(grid[y][x])


def normalize_components(components: list[tuple[int, int, int, int, int]]) -> list[dict[str, int]]:
    sorted_components = sorted(components, key=lambda item: item[0])
    normalized: list[dict[str, int]] = []
    expected_index = 0
    for index, type_code, x, y, rotation in sorted_components:
        if index != expected_index:
            raise ValueError(
                f"Component indices must be dense from 0, expected {expected_index}, got {index}"
            )
        normalized.append(
            {
                "type": int(type_code),
                "x": int(x),
                "y": int(y),
                "rotation": int(rotation),
            }
        )
        expected_index += 1
    return normalized


def discover_ground_raws(
    components: list[dict[str, int]], grid: list[list[int]], width: int, height: int
) -> set[int]:
    ground_raws: set[int] = set()
    for component in components:
        if is_ground_type(component["type"]):
            term_x, term_y = step(
                component["x"], component["y"], opp_dir(component["rotation"])
            )
            raw = region_at(grid, term_x, term_y, width, height)
            if raw != 0:
                ground_raws.add(raw)
    return ground_raws


def build_compaction_map(
    grid: list[list[int]], width: int, height: int, ground_raws: set[int]
) -> dict[int, int]:
    remap: dict[int, int] = {}
    next_node = 1
    for y in range(height):
        for x in range(width):
            raw = int(grid[y][x])
            if raw == 0 or raw in ground_raws or raw in remap:
                continue
            remap[raw] = next_node
            next_node += 1
    return remap


def extract_reference(
    components: list[dict[str, int]], grid: list[list[int]], width: int, height: int
) -> tuple[list[int], list[int], set[int], dict[int, int]]:
    ground_raws = discover_ground_raws(components, grid, width, height)
    remap = build_compaction_map(grid, width, height, ground_raws)
    node0 = [0 for _ in components]
    node1 = [0 for _ in components]

    for idx, component in enumerate(components):
        type_code = component["type"]
        if not is_two_terminal_type(type_code):
            continue

        rotation = component["rotation"]
        anchor_x = component["x"]
        anchor_y = component["y"]

        term0_x, term0_y = step(anchor_x, anchor_y, opp_dir(rotation))
        term1_x, term1_y = step(anchor_x, anchor_y, rotation)
        term1_x, term1_y = step(term1_x, term1_y, rotation)

        raw0 = region_at(grid, term0_x, term0_y, width, height)
        raw1 = region_at(grid, term1_x, term1_y, width, height)

        node0[idx] = 0 if raw0 in ground_raws else remap.get(raw0, 0)
        node1[idx] = 0 if raw1 in ground_raws else remap.get(raw1, 0)

    return node0, node1, ground_raws, remap


def fetchComponentType(tb, idx):
    return tb.state["components"][idx]["type"]


def fetchAnchorPositionX(tb, idx):
    return tb.state["components"][idx]["x"]


def fetchAnchorPositionY(tb, idx):
    return tb.state["components"][idx]["y"]


def fetchComponentRotation(tb, idx):
    return tb.state["components"][idx]["rotation"]


def fetchR(tb, i, j):
    return tb.state["result_grid"][j][i]


def storeNode0(tb, idx, node_i):
    tb.state["node0"][idx] = node_i


def storeNode1(tb, idx, node_i):
    tb.state["node1"][idx] = node_i


def is_ground_component_comb(tb, t):
    return is_ground_type(t)


def is_two_terminal_component_comb(tb, t):
    return is_two_terminal_type(t)


def get_opp_dir_comb(tb, d):
    return opp_dir(d)


def get_nxt_i_comb(tb, i, d):
    next_x, _ = step(i, 0, d)
    return next_x


def get_nxt_j_comb(tb, j, d):
    _, next_y = step(0, j, d)
    return next_y


def extract_registry() -> PrimitiveRTLRegistry:
    return PrimitiveRTLRegistry(
        [
            PrimitiveRTLSpec(name="fetchComponentType", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchAnchorPositionX", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchAnchorPositionY", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchComponentRotation", ports=("idx",), result_port="result", latency=1),
            PrimitiveRTLSpec(name="fetchR", ports=("i", "j"), result_port="result", latency=1),
            PrimitiveRTLSpec(name="storeNode0", ports=("idx", "node_i"), latency=1),
            PrimitiveRTLSpec(name="storeNode1", ports=("idx", "node_i"), latency=1),
            PrimitiveRTLSpec(name="is_ground_component_comb", ports=("t",)),
            PrimitiveRTLSpec(name="is_two_terminal_component_comb", ports=("t",)),
            PrimitiveRTLSpec(name="get_opp_dir_comb", ports=("d",)),
            PrimitiveRTLSpec(name="get_nxt_i_comb", ports=("i", "d")),
            PrimitiveRTLSpec(name="get_nxt_j_comb", ports=("j", "d")),
        ]
    )


def extract_primitives() -> dict[str, PrimitiveModel]:
    return {
        "fetchComponentType": PrimitiveModel("fetchComponentType", fetchComponentType, latency=1),
        "fetchAnchorPositionX": PrimitiveModel("fetchAnchorPositionX", fetchAnchorPositionX, latency=1),
        "fetchAnchorPositionY": PrimitiveModel("fetchAnchorPositionY", fetchAnchorPositionY, latency=1),
        "fetchComponentRotation": PrimitiveModel(
            "fetchComponentRotation", fetchComponentRotation, latency=1
        ),
        "fetchR": PrimitiveModel("fetchR", fetchR, latency=1),
        "storeNode0": PrimitiveModel("storeNode0", storeNode0, latency=1),
        "storeNode1": PrimitiveModel("storeNode1", storeNode1, latency=1),
        "is_ground_component_comb": PrimitiveModel(
            "is_ground_component_comb", is_ground_component_comb
        ),
        "is_two_terminal_component_comb": PrimitiveModel(
            "is_two_terminal_component_comb", is_two_terminal_component_comb
        ),
        "get_opp_dir_comb": PrimitiveModel("get_opp_dir_comb", get_opp_dir_comb),
        "get_nxt_i_comb": PrimitiveModel("get_nxt_i_comb", get_nxt_i_comb),
        "get_nxt_j_comb": PrimitiveModel("get_nxt_j_comb", get_nxt_j_comb),
    }


def make_harness(
    width: int, height: int, components: list[tuple[int, int, int, int, int]], grid: list[list[int]]
) -> SimulationHarness:
    normalized_components = normalize_components(components)
    return SimulationHarness(
        params={
            "par_elem_n": len(normalized_components),
            "grid_height": height,
            "grid_width": width,
        },
        primitives=extract_primitives(),
        initial_state={
            "components": normalized_components,
            "result_grid": [list(map(int, row)) for row in grid],
            "node0": [-1 for _ in normalized_components],
            "node1": [-1 for _ in normalized_components],
        },
    )


class ExtractComponentNodesTests(unittest.TestCase):
    def check_case(
        self,
        *,
        width: int,
        height: int,
        components: list[tuple[int, int, int, int, int]],
        grid: list[list[int]],
    ) -> None:
        normalized_components = normalize_components(components)
        expected_node0, expected_node1, ground_raws, remap = extract_reference(
            normalized_components, grid, width, height
        )

        report = run_all(EXTRACT_COMPONENT_NODES, make_harness(width, height, components, grid))
        self.assertTrue(report.ok, report.mismatches)

        for result in (report.python_result, report.hir_result, report.lir_result):
            self.assertEqual(result.final_state["node0"], expected_node0)
            self.assertEqual(result.final_state["node1"], expected_node1)

        trace_names = [entry.name for entry in report.python_result.trace]
        self.assertIn("fetchComponentType", trace_names)
        self.assertIn("fetchR", trace_names)
        self.assertIn("storeNode0", trace_names)
        self.assertIn("storeNode1", trace_names)
        if any(is_ground_type(component["type"]) for component in normalized_components):
            self.assertIn("is_ground_component_comb", trace_names)
        if any(is_two_terminal_type(component["type"]) for component in normalized_components):
            self.assertIn("is_two_terminal_component_comb", trace_names)

        # Sanity-check the reference side too so the expected values are easy to trust.
        self.assertIsInstance(ground_raws, set)
        self.assertIsInstance(remap, dict)

    def test_compacts_sparse_regions_and_forces_ground_to_zero(self) -> None:
        self.check_case(
            width=5,
            height=4,
            components=[
                (0, TYPE_GROUND, 0, 2, 1),
                (1, TYPE_RL, 2, 1, 2),
                (2, TYPE_VL, 1, 1, 1),
            ],
            grid=[
                [0, 0, 0, 0, 0],
                [7, 0, 0, 11, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        )

    def test_handles_all_rotations_for_two_terminal_components(self) -> None:
        self.check_case(
            width=7,
            height=7,
            components=[
                (0, TYPE_RL, 3, 2, 1),
                (1, TYPE_VL, 2, 3, 0),
                (2, TYPE_IL, 3, 4, 3),
                (3, TYPE_CL, 4, 3, 2),
            ],
            grid=[
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 13, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 14, 0, 0, 0, 16, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 15, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        )

    def test_handles_shared_regions_and_missing_connections(self) -> None:
        self.check_case(
            width=6,
            height=5,
            components=[
                (0, TYPE_GROUND, 1, 3, 1),
                (1, TYPE_RR, 2, 2, 2),
                (2, TYPE_LR, 3, 1, 1),
                (3, TYPE_CR, 4, 2, 0),
            ],
            grid=[
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [5, 0, 0, 0, 17, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        )

    def test_treats_multiple_grounded_regions_as_node_zero(self) -> None:
        self.check_case(
            width=6,
            height=4,
            components=[
                (0, TYPE_GROUND, 0, 2, 1),
                (1, TYPE_GROUND, 4, 2, 1),
                (2, TYPE_RL, 2, 1, 2),
                (3, TYPE_CL, 2, 1, 1),
            ],
            grid=[
                [0, 0, 0, 0, 0, 0],
                [7, 0, 0, 11, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 25, 0, 0, 0],
            ],
        )

    def test_compiles_with_expected_component_and_region_interfaces(self) -> None:
        artifact = compile_file(EXTRACT_COMPONENT_NODES_PATH, extract_registry())

        self.assertEqual(artifact.report.function_name, "extract_component_nodes")
        self.assertIn("module extract_component_nodes", artifact.verilog)
        self.assertIn("fetchComponentType_start", artifact.verilog)
        self.assertIn("fetchAnchorPositionX_start", artifact.verilog)
        self.assertIn("fetchAnchorPositionY_start", artifact.verilog)
        self.assertIn("fetchComponentRotation_start", artifact.verilog)
        self.assertIn("fetchR_start", artifact.verilog)
        self.assertIn("storeNode0_start", artifact.verilog)
        self.assertIn("storeNode1_start", artifact.verilog)
        self.assertIn("is_ground_component_comb(", artifact.verilog)
        self.assertIn("is_two_terminal_component_comb(", artifact.verilog)
        self.assertIn("get_opp_dir_comb(", artifact.verilog)
        self.assertIn("get_nxt_i_comb(", artifact.verilog)
        self.assertIn("get_nxt_j_comb(", artifact.verilog)

        primitive_names = {item.name for item in artifact.report.primitive_usage}
        self.assertIn("fetchComponentType", primitive_names)
        self.assertIn("fetchComponentRotation", primitive_names)
        self.assertIn("fetchR", primitive_names)
        self.assertIn("storeNode0", primitive_names)
        self.assertIn("storeNode1", primitive_names)


if __name__ == "__main__":
    unittest.main()
