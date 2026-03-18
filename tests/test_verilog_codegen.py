import unittest
from pathlib import Path

from compiler.ast_to_hir import lower_source
from compiler.hir_to_lir import lower_func
from compiler.lir import Await, StartOp
from compiler.lir_to_verilog_model import RTLBranchTransition, RTLModuleConfig, RTLReturnTransition, RTLWaitTransition, lower_to_verilog_model
from compiler.primitive_rtl import PrimitiveRTLRegistry, PrimitiveRTLSpec
from compiler.verilog_codegen import generate_verilog


ROOT = Path(__file__).resolve().parents[1]
LU_CORE = (ROOT / "examples" / "lu_core.dsl.py").read_text()


def lower_to_model(source: str, registry: PrimitiveRTLRegistry, config: RTLModuleConfig | None = None):
    return lower_to_verilog_model(lower_func(lower_source(source)), registry, config)


def default_registry() -> PrimitiveRTLRegistry:
    return PrimitiveRTLRegistry(
        [
            PrimitiveRTLSpec(name="add1_comb", ports=("v",)),
            PrimitiveRTLSpec(name="neg_comb", ports=("v",)),
            PrimitiveRTLSpec(name="touch_comb", ports=("v",)),
            PrimitiveRTLSpec(name="load", ports=("i",), result_port="result", latency=3),
            PrimitiveRTLSpec(name="store", ports=("i", "v"), latency=1),
            PrimitiveRTLSpec(name="fetch_A", ports=("i", "j"), result_port="result", latency=2),
            PrimitiveRTLSpec(name="fetch_LU", ports=("i", "j"), result_port="result", latency=2),
            PrimitiveRTLSpec(name="store_LU", ports=("i", "j", "v"), latency=1),
            PrimitiveRTLSpec(name="fma", ports=("a", "b", "c"), result_port="result", latency=3),
            PrimitiveRTLSpec(name="div", ports=("a", "b"), result_port="result", latency=4),
        ]
    )


class VerilogBackendTests(unittest.TestCase):
    def test_view_model_lowers_if_branch_kernel(self) -> None:
        source = '''
def branchy(par_n):
    u8_i = 0
    if u8_i < par_n:
        u8_i = u8_i + 1
    else:
        u8_i = u8_i - 1
    return u8_i
'''
        module = lower_to_model(source, default_registry())
        state_map = {state.lir_label: state for state in module.states}

        self.assertIn("entry", state_map)
        self.assertEqual(module.entry_state, "S_ENTRY")
        self.assertIsInstance(state_map["entry"].transition, RTLBranchTransition)
        self.assertIn("line 3: u8_i = 0", state_map["entry"].comment.lines[1])
        self.assertIsInstance(state_map["if_end_1"].transition, RTLReturnTransition)

    def test_view_model_lowers_wait_state_for_blocking_call(self) -> None:
        source = '''
def io_kernel(par_n):
    u8_i = 0
    u8_x = 0
    u8_x = load(i=u8_i)
    return u8_x
'''
        module = lower_to_model(source, default_registry())
        state_map = {state.lir_label: state for state in module.states}

        self.assertIsInstance(state_map["entry"].transition, RTLWaitTransition)
        self.assertEqual(state_map["entry"].transition.done_signal, "load_done")
        self.assertEqual(state_map["entry"].primitive_start.start_signal, "load_start")
        self.assertEqual(state_map["entry"].primitive_start.result_signal, "load_result")

    def test_view_model_keeps_hidden_for_index_internal(self) -> None:
        source = '''
def has_for(par_n):
    u8_i = 0
    u8_sum = 0
    for u8_i in range(par_n):
        u8_sum = u8_sum + u8_i
    return u8_sum
'''
        module = lower_to_model(source, default_registry())

        self.assertTrue(any(reg.name.startswith("__for_idx_") for reg in module.temps))
        self.assertFalse(any(port.name.startswith("__for_idx_") for port in module.ports))

    def test_generated_verilog_contains_module_skeleton_and_comments(self) -> None:
        source = '''
def branchy(par_n):
    u8_i = 0
    if u8_i < par_n:
        u8_i = u8_i + 1
    else:
        u8_i = u8_i - 1
    return u8_i
'''
        verilog = generate_verilog(lower_func(lower_source(source)), default_registry())

        self.assertIn("module branchy", verilog)
        self.assertIn("always_ff @(posedge clk or negedge rst_n)", verilog)
        self.assertIn("always_comb begin", verilog)
        self.assertIn("typedef enum logic", verilog)
        self.assertIn("// LIR block: entry", verilog)
        self.assertIn("// line 3: u8_i = 0", verilog)
        self.assertIn("S_IDLE", verilog)
        self.assertIn("S_DONE", verilog)
        self.assertNotIn("endend", verilog)
        self.assertNotIn("endcaseend", verilog)
        self.assertNotIn("endmodulemodule", verilog)

    def test_generated_verilog_distinguishes_comb_and_blocking_primitives(self) -> None:
        source = '''
def seq(par_n):
    u8_i = 0
    u8_x = 0
    touch_comb(v=u8_i)
    u8_x = add1_comb(v=u8_i)
    u8_x = load(i=u8_i)
    store(i=u8_i, v=u8_x)
    return u8_x
'''
        verilog = generate_verilog(lower_func(lower_source(source)), default_registry())

        self.assertIn("load_start = 1'b1;", verilog)
        self.assertIn("store_start = 1'b1;", verilog)
        self.assertIn("add1_comb(u8_i)", verilog)
        self.assertNotIn("add1_comb_start", verilog)
        self.assertNotIn("touch_comb_start", verilog)

    def test_generated_verilog_for_lu_core_is_deterministic_and_comment_rich(self) -> None:
        registry = default_registry()
        func_lir = lower_func(lower_source(LU_CORE))
        verilog_a, debug_a = generate_verilog(func_lir, registry, include_debug=True)
        verilog_b, debug_b = generate_verilog(func_lir, registry, include_debug=True)

        self.assertEqual(verilog_a, verilog_b)
        self.assertEqual(debug_a, debug_b)
        self.assertIn("module lu_core", verilog_a)
        self.assertIn("// LIR block: for_header_0", verilog_a)
        self.assertIn("fetch_A_start = 1'b1;", verilog_a)
        self.assertIn("fetch_LU_start = 1'b1;", verilog_a)
        self.assertIn("fma_start = 1'b1;", verilog_a)
        self.assertIn("div_start = 1'b1;", verilog_a)
        self.assertIn("store_LU_start = 1'b1;", verilog_a)
        self.assertIn("neg_comb(", verilog_a)
        self.assertNotIn("endend", verilog_a)
        self.assertNotIn("endcaseend", verilog_a)
        self.assertGreater(debug_a["wait_state_count"], 0)

    def test_wait_state_count_matches_lir_await_count(self) -> None:
        source = '''
def seq(par_n):
    u8_i = 0
    u8_x = 0
    u8_x = load(i=u8_i)
    store(i=u8_i, v=u8_x)
    return u8_x
'''
        func_lir = lower_func(lower_source(source))
        await_count = sum(
            1 for block in func_lir.blocks.values() if isinstance(block.term, Await)
        )
        start_count = sum(
            1 for block in func_lir.blocks.values() for op in block.ops if isinstance(op, StartOp)
        )
        _, debug = generate_verilog(func_lir, default_registry(), include_debug=True)

        self.assertEqual(await_count, start_count)
        self.assertEqual(debug["wait_state_count"], await_count)


if __name__ == "__main__":
    unittest.main()
