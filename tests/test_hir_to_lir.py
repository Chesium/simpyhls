import unittest
from pathlib import Path

from compiler.ast_to_hir import lower_source
from compiler.hir import Call, VarRef
from compiler.hir_to_lir import HIRToLIRError, lower_func
from compiler.lir import AssignOp, Await, Branch, Jump, Return, SideEffectOp, StartOp, format_func_lir


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "lu_core.dsl.py"


def lower_to_lir(source: str):
    return lower_func(lower_source(source))


class HIRToLIRLoweringTests(unittest.TestCase):
    def test_lower_straight_line_code_and_blocking_calls(self) -> None:
        source = '''
def seq(par_n):
    u8_i = 0
    f32_acc = 0
    u8_i = u8_i + 1
    touch_comb(v=u8_i)
    f32_acc = neg_comb(v=f32_acc)
    f32_acc = load(i=u8_i)
    store(i=u8_i, v=f32_acc)
    return f32_acc
'''
        func_lir = lower_to_lir(source)

        self.assertEqual(list(func_lir.blocks), ["entry", "after_call_0", "after_call_1"])

        entry = func_lir.blocks["entry"]
        self.assertEqual(len(entry.ops), 6)
        self.assertTrue(all(isinstance(op, AssignOp) for op in entry.ops[:3]))
        self.assertIsInstance(entry.ops[3], SideEffectOp)
        self.assertIsInstance(entry.ops[3].call, Call)
        self.assertEqual(entry.ops[3].call.func, "touch_comb")
        self.assertIsInstance(entry.ops[4], AssignOp)
        self.assertIsInstance(entry.ops[4].value, Call)
        self.assertEqual(entry.ops[4].value.func, "neg_comb")
        self.assertIsInstance(entry.ops[5], StartOp)
        self.assertEqual(entry.ops[5].call.func, "load")
        self.assertEqual(entry.ops[5].result.name, "f32_acc")
        self.assertEqual(entry.ops[5].token, "op0")
        self.assertIsInstance(entry.term, Await)
        self.assertEqual(entry.term.target, "after_call_0")
        self.assertEqual(entry.term.token, "op0")
        self.assertIsNotNone(entry.term.source_info)

        after_load = func_lir.blocks["after_call_0"]
        self.assertEqual(len(after_load.ops), 1)
        self.assertIsInstance(after_load.ops[0], StartOp)
        self.assertEqual(after_load.ops[0].call.func, "store")
        self.assertIsNone(after_load.ops[0].result)
        self.assertEqual(after_load.ops[0].token, "op1")
        self.assertIsInstance(after_load.term, Await)
        self.assertEqual(after_load.term.target, "after_call_1")
        self.assertEqual(after_load.term.token, "op1")

        exit_block = func_lir.blocks["after_call_1"]
        self.assertEqual(exit_block.ops, [])
        self.assertIsInstance(exit_block.term, Return)
        self.assertIsInstance(exit_block.term.value, VarRef)
        self.assertEqual(exit_block.term.value.var.name, "f32_acc")

    def test_lower_if_to_branch_then_else_and_merge(self) -> None:
        source = '''
def branchy(par_n):
    u8_i = 0
    if u8_i < par_n:
        u8_i = u8_i + 1
    else:
        u8_i = u8_i - 1
    return u8_i
'''
        func_lir = lower_to_lir(source)

        rendered = format_func_lir(func_lir)
        self.assertIn("lir func branchy(par_n)", rendered)
        self.assertIn("entry:", rendered)
        self.assertIn("# line 3: u8_i = 0", rendered)
        self.assertIn("branch (u8_i < par_n) ? if_then_0 : if_else_2", rendered)
        self.assertIn("# line 8: return u8_i", rendered)
        self.assertIn("# line 7: u8_i = u8_i - 1", rendered)

        self.assertIsInstance(func_lir.blocks["entry"].term, Branch)
        self.assertIsInstance(func_lir.blocks["if_then_0"].term, Jump)
        self.assertEqual(func_lir.blocks["if_then_0"].term.target, "if_end_1")
        self.assertIsInstance(func_lir.blocks["if_else_2"].term, Jump)
        self.assertEqual(func_lir.blocks["if_else_2"].term.target, "if_end_1")

    def test_lower_while_loop_with_blocking_body_op(self) -> None:
        source = '''
def loop(par_n):
    u8_i = 0
    f32_acc = 0
    while u8_i < par_n:
        f32_acc = load(i=u8_i)
        u8_i = u8_i + 1
    return u8_i
'''
        func_lir = lower_to_lir(source)

        self.assertEqual(
            list(func_lir.blocks),
            ["entry", "while_header_0", "while_body_1", "while_end_2", "after_call_3"],
        )

        entry = func_lir.blocks["entry"]
        self.assertTrue(all(isinstance(op, AssignOp) for op in entry.ops))
        self.assertIsInstance(entry.term, Jump)
        self.assertEqual(entry.term.target, "while_header_0")

        header = func_lir.blocks["while_header_0"]
        self.assertIsInstance(header.term, Branch)
        self.assertEqual(header.term.true_target, "while_body_1")
        self.assertEqual(header.term.false_target, "while_end_2")

        body = func_lir.blocks["while_body_1"]
        self.assertEqual(len(body.ops), 1)
        self.assertIsInstance(body.ops[0], StartOp)
        self.assertEqual(body.ops[0].call.func, "load")
        self.assertEqual(body.ops[0].token, "op0")
        self.assertIsInstance(body.term, Await)
        self.assertEqual(body.term.target, "after_call_3")
        self.assertEqual(body.term.token, "op0")

        after_call = func_lir.blocks["after_call_3"]
        self.assertEqual(len(after_call.ops), 1)
        self.assertIsInstance(after_call.ops[0], AssignOp)
        self.assertIsInstance(after_call.term, Jump)
        self.assertEqual(after_call.term.target, "while_header_0")

        exit_block = func_lir.blocks["while_end_2"]
        self.assertIsInstance(exit_block.term, Return)
        self.assertIsInstance(exit_block.term.value, VarRef)
        self.assertEqual(exit_block.term.value.var.name, "u8_i")

    def test_lower_for_loop_with_explicit_bounds_and_blocking_body_op(self) -> None:
        source = '''
def has_for(par_n):
    u8_i = 0
    f32_acc = 0
    for u8_i in range(1, par_n):
        f32_acc = load(i=u8_i)
    return u8_i
'''
        func_lir = lower_to_lir(source)

        self.assertEqual(
            list(func_lir.blocks),
            ["entry", "for_header_0", "for_body_1", "for_end_2", "after_call_3"],
        )

        entry = func_lir.blocks["entry"]
        self.assertEqual(len(entry.ops), 3)
        self.assertTrue(all(isinstance(op, AssignOp) for op in entry.ops))
        self.assertTrue(entry.ops[2].target.name.startswith("__for_idx_"))
        self.assertIsInstance(entry.term, Jump)
        self.assertEqual(entry.term.target, "for_header_0")

        header = func_lir.blocks["for_header_0"]
        self.assertIsInstance(header.term, Branch)
        self.assertEqual(header.term.true_target, "for_body_1")
        self.assertEqual(header.term.false_target, "for_end_2")

        body = func_lir.blocks["for_body_1"]
        self.assertEqual(len(body.ops), 2)
        self.assertIsInstance(body.ops[0], AssignOp)
        self.assertEqual(body.ops[0].target.name, "u8_i")
        self.assertIsInstance(body.ops[1], StartOp)
        self.assertEqual(body.ops[1].call.func, "load")
        self.assertIsInstance(body.term, Await)
        self.assertEqual(body.term.target, "after_call_3")
        self.assertEqual(body.term.token, "op0")

        latch = func_lir.blocks["after_call_3"]
        self.assertEqual(len(latch.ops), 1)
        self.assertIsInstance(latch.ops[0], AssignOp)
        self.assertTrue(latch.ops[0].target.name.startswith("__for_idx_"))
        self.assertIsInstance(latch.term, Jump)
        self.assertEqual(latch.term.target, "for_header_0")

        exit_block = func_lir.blocks["for_end_2"]
        self.assertIsInstance(exit_block.term, Return)
        self.assertIsInstance(exit_block.term.value, VarRef)
        self.assertEqual(exit_block.term.value.var.name, "u8_i")

    def test_lower_lu_core_example_to_lir(self) -> None:
        func_lir = lower_func(lower_source(EXAMPLE.read_text()))
        rendered = format_func_lir(func_lir)

        self.assertEqual(func_lir.name, "lu_core")
        self.assertEqual(func_lir.entry, "entry")
        self.assertIn("for_header_", rendered)
        self.assertIn("for_body_", rendered)
        self.assertIn("if_then_", rendered)
        self.assertIn("branch (u8_i > u8_j) ?", rendered)
        self.assertIn("start f32_1 = fetch_A(i=u8_i, j=u8_j) [op0]", rendered)
        self.assertIn("start f32_2 = fetch_LU(i=u8_i, j=u8_k)", rendered)
        self.assertIn("start f32_1 = fma(a=f32_2, b=f32_3, c=f32_1)", rendered)
        self.assertIn("start store_LU(i=u8_i, j=u8_j, v=f32_1)", rendered)
        self.assertIn("await ->", rendered)
        self.assertIn("return", rendered)

        start_ops = [
            op
            for block in func_lir.blocks.values()
            for op in block.ops
            if isinstance(op, StartOp)
        ]
        self.assertGreaterEqual(len(start_ops), 6)
        self.assertTrue(any(op.call.func == "store_LU" and op.result is None for op in start_ops))
        self.assertFalse(any(op.call.func == "neg_comb" for op in start_ops))
        self.assertTrue(any(isinstance(block.term, Return) for block in func_lir.blocks.values()))

    def test_blocking_call_in_condition_is_rejected(self) -> None:
        source = '''
def bad(par_n):
    u8_i = 0
    while ready(i=u8_i):
        u8_i = u8_i + 1
    return u8_i
'''
        with self.assertRaises(HIRToLIRError) as ctx:
            lower_to_lir(source)

        self.assertIn("Blocking primitive call 'ready'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
