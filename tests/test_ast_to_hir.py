import unittest
from pathlib import Path

from compiler.ast_checker import DSLCheckError
from compiler.ast_to_hir import lower_source
from compiler.hir import Assign, BinOp, Call, Compare, Const, ExprStmt, ForRangeStmt, IfStmt, ReturnStmt, Select, VarRef, WhileStmt


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples' / 'lu_core.dsl.py'


class ASTToHIRLoweringTests(unittest.TestCase):
    def test_lower_lu_core_preserves_nested_control_flow(self) -> None:
        func_ir = lower_source(EXAMPLE.read_text())

        self.assertEqual(func_ir.name, 'lu_core')
        self.assertEqual([arg.name for arg in func_ir.args], ['par_n'])
        self.assertEqual(func_ir.args[0].typ.name, 'i32')
        self.assertEqual(
            [var.name for var in func_ir.locals],
            ['f32_1', 'f32_2', 'f32_3', 'u8_i', 'u8_j', 'u8_k', 'u8_m'],
        )
        self.assertEqual(
            [var.typ.name for var in func_ir.locals],
            ['f32', 'f32', 'f32', 'u8', 'u8', 'u8', 'u8'],
        )

        init_stmts = func_ir.body[:7]
        self.assertTrue(all(isinstance(stmt, Assign) for stmt in init_stmts))
        self.assertEqual(init_stmts[0].target.typ.name, 'f32')
        self.assertIsInstance(init_stmts[0].value, Const)
        self.assertEqual(init_stmts[0].value.typ.name, 'f32')

        outer_loop = func_ir.body[7]
        self.assertIsInstance(outer_loop, ForRangeStmt)
        self.assertEqual(outer_loop.iter_var.name, 'u8_j')
        self.assertIsInstance(outer_loop.start, Const)
        self.assertEqual(outer_loop.start.value, 0)
        self.assertEqual(outer_loop.start.typ.name, 'u8')
        self.assertIsInstance(outer_loop.stop, VarRef)
        self.assertEqual(outer_loop.stop.var.name, 'par_n')

        inner_loop = outer_loop.body[0]
        self.assertIsInstance(inner_loop, ForRangeStmt)
        self.assertEqual(len(inner_loop.body), 7)

        fetch_assign = inner_loop.body[0]
        self.assertIsInstance(fetch_assign, Assign)
        self.assertIsInstance(fetch_assign.value, Call)
        self.assertEqual(fetch_assign.value.func, 'fetch_A')
        self.assertEqual([arg.name for arg in fetch_assign.value.args], ['i', 'j'])
        self.assertEqual(
            [arg.value.var.name for arg in fetch_assign.value.args if isinstance(arg.value, VarRef)],
            ['u8_i', 'u8_j'],
        )

        select_assign = inner_loop.body[2]
        self.assertIsInstance(select_assign, Assign)
        self.assertIsInstance(select_assign.value, Select)

        reduction_loop = inner_loop.body[3]
        self.assertIsInstance(reduction_loop, ForRangeStmt)
        self.assertEqual(reduction_loop.iter_var.name, 'u8_k')

        guard = inner_loop.body[4]
        self.assertIsInstance(guard, IfStmt)
        self.assertIsInstance(guard.cond, Compare)
        self.assertEqual(guard.cond.op, '>')

        store_stmt = inner_loop.body[6]
        self.assertIsInstance(store_stmt, ExprStmt)
        self.assertIsInstance(store_stmt.value, Call)
        self.assertEqual(store_stmt.value.func, 'store_LU')
        self.assertEqual([arg.name for arg in store_stmt.value.args], ['i', 'j', 'v'])

    def test_lower_bool_ops_and_explicit_range_bounds(self) -> None:
        source = '''
def ctrl(par_n):
    u8_i = 0
    u8_sum = 0
    for u8_i in range(1, par_n):
        if u8_i < par_n and u8_sum < par_n:
            u8_sum = u8_sum + u8_i
        else:
            u8_sum = u8_sum - 1
    return u8_sum
'''
        func_ir = lower_source(source)

        loop_stmt = func_ir.body[2]
        self.assertIsInstance(loop_stmt, ForRangeStmt)
        self.assertIsInstance(loop_stmt.start, Const)
        self.assertEqual(loop_stmt.start.value, 1)
        self.assertEqual(loop_stmt.start.typ.name, 'u8')

        guard = loop_stmt.body[0]
        self.assertIsInstance(guard, IfStmt)
        self.assertIsInstance(guard.cond, BinOp)
        self.assertEqual(guard.cond.op, 'and')
        self.assertIsInstance(guard.cond.lhs, Compare)
        self.assertIsInstance(guard.cond.rhs, Compare)

        then_assign = guard.then_body[0]
        self.assertIsInstance(then_assign, Assign)
        self.assertIsInstance(then_assign.value, BinOp)
        self.assertEqual(then_assign.value.op, '+')

        else_assign = guard.else_body[0]
        self.assertIsInstance(else_assign, Assign)
        self.assertIsInstance(else_assign.value, BinOp)
        self.assertEqual(else_assign.value.op, '-')

        ret = func_ir.body[3]
        self.assertIsInstance(ret, ReturnStmt)
        self.assertIsInstance(ret.value, VarRef)
        self.assertEqual(ret.value.var.name, 'u8_sum')

    def test_lower_while_select_and_primitive_expr_stmt(self) -> None:
        source = '''
def runner(par_n):
    u8_i = 0
    u8_limit = 0
    while u8_i < par_n:
        if u8_i < par_n:
            tick(i=u8_i)
        else:
            u8_limit = u8_i if u8_i < par_n else par_n
        u8_i = u8_i + 1
    return u8_limit
'''
        func_ir = lower_source(source)

        loop_stmt = func_ir.body[2]
        self.assertIsInstance(loop_stmt, WhileStmt)
        self.assertIsInstance(loop_stmt.cond, Compare)
        self.assertEqual(loop_stmt.cond.op, '<')

        branch = loop_stmt.body[0]
        self.assertIsInstance(branch, IfStmt)
        self.assertIsInstance(branch.then_body[0], ExprStmt)
        self.assertIsInstance(branch.then_body[0].value, Call)
        self.assertEqual(branch.then_body[0].value.func, 'tick')
        self.assertEqual([arg.name for arg in branch.then_body[0].value.args], ['i'])
        self.assertIsInstance(branch.then_body[0].value.args[0].value, VarRef)
        self.assertEqual(branch.then_body[0].value.args[0].value.var.name, 'u8_i')

        else_assign = branch.else_body[0]
        self.assertIsInstance(else_assign, Assign)
        self.assertIsInstance(else_assign.value, Select)

        increment = loop_stmt.body[1]
        self.assertIsInstance(increment, Assign)
        self.assertIsInstance(increment.value, BinOp)
        self.assertEqual(increment.value.op, '+')

    def test_invalid_dsl_is_rejected_before_lowering(self) -> None:
        bad_sources = {
            'non_range_loop': '''
def broken(par_n):
    u8_i = 0
    for u8_i in items:
        u8_i = u8_i + 1
''',
            'undeclared_local': '''
def broken(par_n):
    u8_i = 0
    u8_j = u8_i + 1
''',
            'positional_primitive_args': '''
def broken(par_n):
    u8_i = 0
    u8_i = step(u8_i)
''',
        }

        for name, source in bad_sources.items():
            with self.subTest(name=name):
                with self.assertRaises(DSLCheckError):
                    lower_source(source)


if __name__ == '__main__':
    unittest.main()
