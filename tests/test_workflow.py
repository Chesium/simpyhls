import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from compiler.primitive_rtl import PrimitiveRTLRegistry, PrimitiveRTLSpec
from compiler.verilog_codegen import generate_verilog
from compiler.workflow import compile_file, compile_source, format_codegen_report, write_compilation_outputs


ROOT = Path(__file__).resolve().parents[1]
LU_CORE_PATH = ROOT / "examples" / "lu_core.dsl.py"


def default_registry() -> PrimitiveRTLRegistry:
    return PrimitiveRTLRegistry(
        [
            PrimitiveRTLSpec(name="add1_comb", ports=("v",)),
            PrimitiveRTLSpec(name="neg_comb", ports=("v",)),
            PrimitiveRTLSpec(name="load", ports=("i",), result_port="result", latency=3),
            PrimitiveRTLSpec(name="fetch_A", ports=("i", "j"), result_port="result", latency=2),
            PrimitiveRTLSpec(name="fetch_LU", ports=("i", "j"), result_port="result", latency=2),
            PrimitiveRTLSpec(name="store_LU", ports=("i", "j", "v"), latency=1),
            PrimitiveRTLSpec(name="fma", ports=("a", "b", "c"), result_port="result", latency=3),
            PrimitiveRTLSpec(name="div", ports=("a", "b"), result_port="result", latency=4),
        ]
    )


class WorkflowTests(unittest.TestCase):
    def test_compile_source_produces_verilog_and_exact_fixed_loop_wait_report(self) -> None:
        source = '''
def fixed_loop(par_n):
    u8_i = 0
    u8_x = 0
    for u8_i in range(2):
        u8_x = load(i=u8_i)
    return u8_x
'''
        artifact = compile_source(source, default_registry())

        self.assertIn("module fixed_loop", artifact.verilog)
        self.assertEqual(artifact.verilog, generate_verilog(artifact.lir, default_registry()))
        self.assertEqual(artifact.report.control_flow, {"if": 0, "for": 1, "while": 0})
        self.assertEqual(artifact.report.latency.lower_bound_cycles, 6)
        self.assertEqual(artifact.report.latency.upper_bound_cycles, 6)
        self.assertTrue(artifact.report.latency.exact)
        self.assertEqual(artifact.report.latency.symbolic_expr, "(2) * (3)")
        self.assertEqual(artifact.report.rtl_wait_state_count, 1)
        self.assertEqual(artifact.report.primitive_usage[0].name, "load")
        self.assertEqual(artifact.report.primitive_usage[0].call_sites, 1)

        report_text = format_codegen_report(artifact.report)
        self.assertIn("Blocking wait latency: lower_bound=6, upper_bound=6", report_text)
        self.assertIn("load: kind=blocking, latency=3, call_sites=1", report_text)

        with TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            verilog_path = out_dir / "fixed_loop.sv"
            report_path = out_dir / "fixed_loop.report.txt"
            write_compilation_outputs(artifact, verilog_path, report_path)
            self.assertEqual(verilog_path.read_text(), artifact.verilog)
            self.assertIn("Function: fixed_loop", report_path.read_text())

    def test_compile_file_covers_lu_core_and_reports_dynamic_latency(self) -> None:
        artifact = compile_file(LU_CORE_PATH, default_registry())

        self.assertIn("module lu_core", artifact.verilog)
        self.assertEqual(artifact.report.function_name, "lu_core")
        self.assertEqual(artifact.report.control_flow, {"if": 1, "for": 3, "while": 0})
        self.assertGreater(artifact.report.rtl_state_count, artifact.report.lir_block_count)
        self.assertGreater(artifact.report.rtl_wait_state_count, 0)
        self.assertEqual(artifact.report.latency.lower_bound_cycles, 0)
        self.assertIsNone(artifact.report.latency.upper_bound_cycles)
        self.assertIn("max(par_n, 0)", artifact.report.latency.symbolic_expr)

        primitive_names = {item.name for item in artifact.report.primitive_usage}
        self.assertIn("fetch_A", primitive_names)
        self.assertIn("fetch_LU", primitive_names)
        self.assertIn("fma", primitive_names)
        self.assertIn("div", primitive_names)
        self.assertIn("store_LU", primitive_names)
        self.assertIn("neg_comb", primitive_names)

        report_text = format_codegen_report(artifact.report)
        self.assertIn("Function: lu_core", report_text)
        self.assertIn("Blocking call sites:", report_text)
        self.assertIn("fetch_A", report_text)
        self.assertIn("store_LU", report_text)


if __name__ == "__main__":
    unittest.main()
