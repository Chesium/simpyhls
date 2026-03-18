import ast
from typing import Optional

from .ast_checker import DSLCheckError, DSLFrontendChecker
from .hir import (
    Assign,
    BinOp,
    Call,
    CallArg,
    Compare,
    Const,
    Expr,
    ExprStmt,
    ForRangeStmt,
    FuncIR,
    IfStmt,
    ReturnStmt,
    Select,
    Stmt,
    Type,
    UnaryOp,
    Var,
    VarRef,
    WhileStmt,
)


class ASTLoweringError(Exception):
    pass


_BOOL_TYPE = Type("bool")
_DEFAULT_INT_TYPE = Type("i32")
_UNARY_OPS = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "not",
}
_BIN_OPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}
_BOOL_OPS = {
    ast.And: "and",
    ast.Or: "or",
}
_COMPARE_OPS = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


class ASTToHIRLowerer:
    """Lower a frontend-checked Python AST into the structured HIR in hir.py."""

    def __init__(self) -> None:
        self.source: Optional[str] = None
        self._checker = DSLFrontendChecker()
        self._vars: dict[str, Var] = {}

    def lower_source(self, source: str) -> FuncIR:
        tree = ast.parse(source)
        return self.lower_tree(tree, source=source)

    def lower_tree(self, tree: ast.AST, source: Optional[str] = None) -> FuncIR:
        if source is not None:
            self.source = source

        info = self._checker.check_tree(tree, source=source)
        if not isinstance(tree, ast.Module) or len(tree.body) != 1:
            raise ASTLoweringError("Expected a module with exactly one top-level function")

        func = tree.body[0]
        if not isinstance(func, ast.FunctionDef):
            raise ASTLoweringError("Expected a top-level function definition")

        self._vars = {}
        arg_vars = [self._declare_var(name, is_param=True) for name in info.params]
        local_vars = [self._declare_var(name, is_param=False) for name in info.locals]

        decl_stmts = []
        body_start = 0
        while body_start < len(func.body) and self._is_decl_assign(func.body[body_start]):
            decl_stmts.append(self._lower_assign(func.body[body_start]))
            body_start += 1

        body = decl_stmts + [self._lower_stmt(stmt) for stmt in func.body[body_start:]]
        return FuncIR(name=func.name, args=arg_vars, locals=local_vars, body=body)

    def _declare_var(self, name: str, *, is_param: bool) -> Var:
        var = Var(name=name, typ=self._infer_var_type(name, is_param=is_param))
        self._vars[name] = var
        return var

    def _infer_var_type(self, name: str, *, is_param: bool) -> Type:
        if "_" in name:
            prefix = name.split("_", 1)[0]
            if prefix == "bool" or any(ch.isdigit() for ch in prefix):
                return Type(prefix)
        if is_param:
            return Type(_DEFAULT_INT_TYPE.name)
        return Type(_DEFAULT_INT_TYPE.name)

    def _is_decl_assign(self, stmt: ast.stmt) -> bool:
        return self._checker.is_decl_assign(stmt)

    def _lookup_var(self, name: str, node: ast.AST) -> Var:
        try:
            return self._vars[name]
        except KeyError as exc:
            raise self._err(node, f"Unknown variable '{name}' during lowering") from exc

    def _err(self, node: ast.AST, msg: str) -> ASTLoweringError:
        line = getattr(node, "lineno", "?")
        col = getattr(node, "col_offset", "?")
        seg = ast.get_source_segment(self.source, node) if self.source else None
        if seg:
            return ASTLoweringError(f"[line {line}:{col}] {msg}\n  >>> {seg}")
        return ASTLoweringError(f"[line {line}:{col}] {msg}")

    def _lower_stmt(self, stmt: ast.stmt) -> Stmt:
        if isinstance(stmt, ast.Assign):
            return self._lower_assign(stmt)
        if isinstance(stmt, ast.For):
            return self._lower_for(stmt)
        if isinstance(stmt, ast.While):
            return self._lower_while(stmt)
        if isinstance(stmt, ast.If):
            return self._lower_if(stmt)
        if isinstance(stmt, ast.Expr):
            return ExprStmt(value=self._lower_expr(stmt.value))
        if isinstance(stmt, ast.Return):
            value = self._lower_expr(stmt.value) if stmt.value is not None else None
            return ReturnStmt(value=value)
        raise self._err(stmt, f"Unsupported statement type: {type(stmt).__name__}")

    def _lower_assign(self, node: ast.Assign) -> Assign:
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise self._err(node, "Expected assignment target to be a variable name")

        target_var = self._lookup_var(target.id, target)
        value = self._lower_expr(node.value, expected_type=target_var.typ)
        return Assign(target=target_var, value=value)

    def _lower_for(self, node: ast.For) -> ForRangeStmt:
        if not isinstance(node.target, ast.Name):
            raise self._err(node, "Expected for-loop target to be a variable name")
        iter_var = self._lookup_var(node.target.id, node.target)

        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name):
            raise self._err(node, "Expected for-loop iterator to be range(...)")

        range_args = node.iter.args
        if len(range_args) == 1:
            start = Const(value=0, typ=iter_var.typ)
            stop = self._lower_expr(range_args[0], expected_type=iter_var.typ)
        elif len(range_args) == 2:
            start = self._lower_expr(range_args[0], expected_type=iter_var.typ)
            stop = self._lower_expr(range_args[1], expected_type=iter_var.typ)
        else:
            raise self._err(node.iter, "Expected range(stop) or range(start, stop)")

        body = [self._lower_stmt(stmt) for stmt in node.body]
        return ForRangeStmt(iter_var=iter_var, start=start, stop=stop, body=body)

    def _lower_while(self, node: ast.While) -> WhileStmt:
        cond = self._lower_expr(node.test, expected_type=_BOOL_TYPE)
        body = [self._lower_stmt(stmt) for stmt in node.body]
        return WhileStmt(cond=cond, body=body)

    def _lower_if(self, node: ast.If) -> IfStmt:
        cond = self._lower_expr(node.test, expected_type=_BOOL_TYPE)
        then_body = [self._lower_stmt(stmt) for stmt in node.body]
        else_body = [self._lower_stmt(stmt) for stmt in node.orelse]
        return IfStmt(cond=cond, then_body=then_body, else_body=else_body)

    def _lower_expr(self, expr: ast.AST, expected_type: Optional[Type] = None) -> Expr:
        if isinstance(expr, ast.Name):
            return VarRef(var=self._lookup_var(expr.id, expr))
        if isinstance(expr, ast.Constant):
            return self._lower_constant(expr, expected_type=expected_type)
        if isinstance(expr, ast.UnaryOp):
            return self._lower_unary_op(expr, expected_type=expected_type)
        if isinstance(expr, ast.BinOp):
            return self._lower_bin_op(expr, expected_type=expected_type)
        if isinstance(expr, ast.BoolOp):
            return self._lower_bool_op(expr)
        if isinstance(expr, ast.Compare):
            return self._lower_compare(expr)
        if isinstance(expr, ast.IfExp):
            return Select(
                cond=self._lower_expr(expr.test, expected_type=_BOOL_TYPE),
                true_value=self._lower_expr(expr.body, expected_type=expected_type),
                false_value=self._lower_expr(expr.orelse, expected_type=expected_type),
            )
        if isinstance(expr, ast.Call):
            return self._lower_call(expr)
        raise self._err(expr, f"Unsupported expression type: {type(expr).__name__}")

    def _lower_constant(self, expr: ast.Constant, expected_type: Optional[Type]) -> Const:
        value = expr.value
        if isinstance(value, bool):
            typ = _BOOL_TYPE
        elif expected_type is not None:
            typ = expected_type
        else:
            typ = _DEFAULT_INT_TYPE
        return Const(value=value, typ=Type(typ.name))

    def _lower_unary_op(self, expr: ast.UnaryOp, expected_type: Optional[Type]) -> UnaryOp:
        op = _UNARY_OPS[type(expr.op)]
        operand_type = _BOOL_TYPE if op == "not" else expected_type
        return UnaryOp(op=op, value=self._lower_expr(expr.operand, expected_type=operand_type))

    def _lower_bin_op(self, expr: ast.BinOp, expected_type: Optional[Type]) -> BinOp:
        op = _BIN_OPS[type(expr.op)]
        return BinOp(
            op=op,
            lhs=self._lower_expr(expr.left, expected_type=expected_type),
            rhs=self._lower_expr(expr.right, expected_type=expected_type),
        )

    def _lower_bool_op(self, expr: ast.BoolOp) -> Expr:
        op = _BOOL_OPS[type(expr.op)]
        values = [self._lower_expr(value, expected_type=_BOOL_TYPE) for value in expr.values]
        result = values[0]
        for value in values[1:]:
            result = BinOp(op=op, lhs=result, rhs=value)
        return result

    def _lower_compare(self, expr: ast.Compare) -> Compare:
        op = _COMPARE_OPS[type(expr.ops[0])]
        return Compare(
            op=op,
            lhs=self._lower_expr(expr.left),
            rhs=self._lower_expr(expr.comparators[0]),
        )

    def _lower_call(self, expr: ast.Call) -> Call:
        if not isinstance(expr.func, ast.Name):
            raise self._err(expr, "Expected call target to be a simple name")
        args = [
            CallArg(name=keyword.arg, value=self._lower_expr(keyword.value))
            for keyword in expr.keywords
        ]
        return Call(func=expr.func.id, args=args)


def lower_source(source: str) -> FuncIR:
    return ASTToHIRLowerer().lower_source(source)


def lower_tree(tree: ast.AST, source: Optional[str] = None) -> FuncIR:
    return ASTToHIRLowerer().lower_tree(tree, source=source)


__all__ = [
    "ASTLoweringError",
    "ASTToHIRLowerer",
    "lower_source",
    "lower_tree",
    "DSLCheckError",
]
