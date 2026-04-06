import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


LOCAL_NAME_RE = re.compile(r"^[A-Za-z][0-9]+_[A-Za-z0-9_]+$")


class DSLCheckError(Exception):
    pass


@dataclass
class PrimitiveUse:
    name: str
    is_comb: bool
    call_lines: List[int] = field(default_factory=list)
    observed_ports: Set[str] = field(default_factory=set)


@dataclass
class FrontendInfo:
    function_name: str
    params: List[str]
    locals: Dict[str, int]  # name -> init integer literal
    primitives: Dict[str, PrimitiveUse]


class DSLFrontendChecker(ast.NodeVisitor):
    """
    Frontend checker for the restricted DSL.

    Rules enforced:
    - module contains exactly one top-level function
    - all function params are accepted as integer params
    - all local vars must be declared+initialized at the top:
          Xkk_label = <int-constant>
    - after declaration phase ends, no new locals may appear
    - primitive calls must be direct Name(...) calls with keyword args only
    - primitive names ending in '_comb' are tagged as combinational
    """

    def __init__(self, source: Optional[str] = None):
        self.source = source

        self.func_name: Optional[str] = None
        self.params: List[str] = []
        self.locals: Dict[str, int] = {}
        self.primitives: Dict[str, PrimitiveUse] = {}

        self._in_top_decl_phase = False
        self._seen_non_decl_stmt = False

    # ---------- public API ----------

    def check_source(self, source: str) -> FrontendInfo:
        tree = ast.parse(source)
        return self.check_tree(tree, source=source)

    def check_tree(self, tree: ast.AST, source: Optional[str] = None) -> FrontendInfo:
        if source is not None:
            self.source = source
        self.visit(tree)

        if self.func_name is None:
            raise DSLCheckError("No top-level function found")

        return FrontendInfo(
            function_name=self.func_name,
            params=list(self.params),
            locals=dict(self.locals),
            primitives=self.primitives,
        )

    # ---------- helpers ----------

    def err(self, node: ast.AST, msg: str) -> DSLCheckError:
        line = getattr(node, "lineno", "?")
        col = getattr(node, "col_offset", "?")
        seg = ast.get_source_segment(self.source, node) if self.source else None
        if seg:
            return DSLCheckError(f"[line {line}:{col}] {msg}\n  >>> {seg}")
        return DSLCheckError(f"[line {line}:{col}] {msg}")

    def ensure(self, cond: bool, node: ast.AST, msg: str) -> None:
        if not cond:
            raise self.err(node, msg)

    def is_local_name(self, name: str) -> bool:
        return bool(LOCAL_NAME_RE.match(name))

    def is_decl_assign(self, stmt: ast.stmt) -> bool:
        if not isinstance(stmt, ast.Assign):
            return False
        if len(stmt.targets) != 1:
            return False
        if not isinstance(stmt.targets[0], ast.Name):
            return False
        if not isinstance(stmt.value, ast.Constant):
            return False
        return isinstance(stmt.value.value, int)

    def record_primitive_call(self, node: ast.Call) -> None:
        self.ensure(
            isinstance(node.func, ast.Name),
            node,
            "Primitive call must be direct Name(...)",
        )
        prim_name = node.func.id

        self.ensure(
            len(node.args) == 0, node, "Primitive call must use keyword arguments only"
        )
        self.ensure(
            all(
                isinstance(kw, ast.keyword) and kw.arg is not None
                for kw in node.keywords
            ),
            node,
            "Primitive call cannot use **kwargs",
        )

        seen = set()
        for kw in node.keywords:
            self.ensure(
                kw.arg not in seen,
                node,
                f"Duplicate keyword '{kw.arg}' in primitive call",
            )
            seen.add(kw.arg)
            self.visit_expr(kw.value)

        if prim_name not in self.primitives:
            self.primitives[prim_name] = PrimitiveUse(
                name=prim_name,
                is_comb=prim_name.endswith("_comb"),
            )
            self.primitives[prim_name].observed_ports.update(seen)
        self.primitives[prim_name].call_lines.append(getattr(node, "lineno", -1))

        self.ensure(
            self.primitives[prim_name].observed_ports == seen,
            node,
            "Not consistent port declaration",
        )

    def check_name_load(self, node: ast.Name) -> None:
        allowed = set(self.params) | set(self.locals)
        self.ensure(node.id in allowed, node, f"Use of undeclared name '{node.id}'")

    def check_name_store(self, node: ast.Name) -> None:
        allowed = set(self.locals)
        self.ensure(
            node.id in allowed,
            node,
            f"Assignment introduces undeclared local '{node.id}'",
        )

    # ---------- module / function ----------

    def visit_Module(self, node: ast.Module) -> None:
        funcs = [stmt for stmt in node.body if isinstance(stmt, ast.FunctionDef)]
        self.ensure(
            len(node.body) == 1 and len(funcs) == 1,
            node,
            "Module must contain exactly one top-level function and nothing else",
        )
        self.visit(funcs[0])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.ensure(
            self.func_name is None, node, "Only one top-level function is allowed"
        )
        self.func_name = node.name

        args = node.args
        self.ensure(not args.posonlyargs, node, "posonlyargs are not supported")
        self.ensure(not args.kwonlyargs, node, "kwonlyargs are not supported")
        self.ensure(args.vararg is None, node, "*args is not supported")
        self.ensure(args.kwarg is None, node, "**kwargs is not supported")
        self.ensure(
            not args.defaults, node, "Default parameter values are not supported"
        )
        self.ensure(not args.kw_defaults, node, "Keyword defaults are not supported")

        self.params = [a.arg for a in args.args]
        self.ensure(
            len(list(set(args.args))) == len(args.args),
            node,
            "Duplicate parameter name",
        )

        # declaration phase: contiguous Assign(Name = int-constant) at top
        idx = 0
        while idx < len(node.body) and self.is_decl_assign(node.body[idx]):
            stmt = node.body[idx]
            target = stmt.targets[0]
            value = stmt.value

            assert isinstance(target, ast.Name)
            assert isinstance(value, ast.Constant)
            assert isinstance(value.value, int)

            self.ensure(
                self.is_local_name(target.id),
                target,
                f"Local variable '{target.id}' must match pattern Xkk_label, e.g. f32_acc",
            )
            self.ensure(
                target.id not in self.locals,
                target,
                f"Duplicate local declaration '{target.id}'",
            )
            self.ensure(
                target.id not in self.params,
                target,
                f"Local '{target.id}' conflicts with parameter name",
            )

            self.locals[target.id] = value.value
            idx += 1

        self.ensure(
            len(self.locals) > 0, node, "Function must declare locals at the top"
        )
        self._seen_non_decl_stmt = True

        # Remaining statements are executable body
        for stmt in node.body[idx:]:
            self.visit_stmt(stmt)

    # ---------- statement checking ----------

    def visit_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            self.visit_Assign(stmt)
        elif isinstance(stmt, ast.For):
            self.visit_For(stmt)
        elif isinstance(stmt, ast.While):
            self.visit_While(stmt)
        elif isinstance(stmt, ast.If):
            self.visit_If(stmt)
        elif isinstance(stmt, ast.Expr):
            self.visit_Expr(stmt)
        elif isinstance(stmt, ast.Return):
            self.visit_Return(stmt)
        else:
            raise self.err(stmt, f"Unsupported statement type: {type(stmt).__name__}")

    def visit_Assign(self, node: ast.Assign) -> None:
        self.ensure(
            len(node.targets) == 1, node, "Only single-target assignment is supported"
        )
        target = node.targets[0]
        self.ensure(
            isinstance(target, ast.Name),
            node,
            "Assignment target must be a simple variable name",
        )

        # No new locals after declaration phase
        self.check_name_store(target)
        self.visit_expr(node.value)

    def visit_For(self, node: ast.For) -> None:
        self.ensure(
            isinstance(node.target, ast.Name),
            node,
            "For loop target must be a variable name",
        )
        self.check_name_store(node.target)

        self.ensure(
            isinstance(node.iter, ast.Call),
            node,
            "For loop must iterate over range(...)",
        )
        self.ensure(
            isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range",
            node,
            "Only for ... in range(...) is supported",
        )
        self.ensure(
            len(node.iter.keywords) == 0,
            node,
            "range(...) cannot use keyword arguments",
        )
        self.ensure(
            1 <= len(node.iter.args) <= 2,
            node,
            "Only range(stop) and range(start, stop) are supported",
        )

        for arg in node.iter.args:
            self.visit_expr(arg)

        self.ensure(len(node.orelse) == 0, node, "for ... else is not supported")

        for stmt in node.body:
            self.visit_stmt(stmt)

    def visit_While(self, node: ast.While) -> None:
        self.visit_expr(node.test)
        self.ensure(len(node.orelse) == 0, node, "while ... else is not supported")
        for stmt in node.body:
            self.visit_stmt(stmt)

    def visit_If(self, node: ast.If) -> None:
        self.visit_expr(node.test)
        for stmt in node.body:
            self.visit_stmt(stmt)
        for stmt in node.orelse:
            self.visit_stmt(stmt)

    def visit_Expr(self, node: ast.Expr) -> None:
        # Only primitive calls are allowed as standalone expression statements
        self.ensure(
            isinstance(node.value, ast.Call),
            node,
            "Only primitive calls may appear as standalone expressions",
        )
        self.record_primitive_call(node.value)

    def visit_Return(self, node: ast.Return) -> None:
        # First version: allow bare return or return <expr>
        if node.value is not None:
            self.visit_expr(node.value)

    # ---------- expression checking ----------

    def visit_expr(self, expr: ast.AST) -> None:
        if isinstance(expr, ast.Name):
            self.check_name_load(expr)
        elif isinstance(expr, ast.Constant):
            self.ensure(
                isinstance(expr.value, (int, bool)),
                expr,
                "Only integer/bool constants are supported in this frontend",
            )
        elif isinstance(expr, ast.UnaryOp):
            self.visit_UnaryOp(expr)
        elif isinstance(expr, ast.BinOp):
            self.visit_BinOp(expr)
        elif isinstance(expr, ast.BoolOp):
            self.visit_BoolOp(expr)
        elif isinstance(expr, ast.Compare):
            self.visit_Compare(expr)
        elif isinstance(expr, ast.IfExp):
            self.visit_IfExp(expr)
        elif isinstance(expr, ast.Call):
            self.record_primitive_call(expr)
        else:
            raise self.err(expr, f"Unsupported expression type: {type(expr).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.ensure(
            isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)),
            node,
            "Only unary +, unary -, and not are supported",
        )
        self.visit_expr(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.ensure(
            isinstance(
                node.op,
                (
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.BitAnd,
                    ast.BitOr,
                    ast.BitXor,
                    ast.LShift,
                    ast.RShift,
                ),
            ),
            node,
            "Only +, -, *, &, |, ^, <<, >> are supported",
        )
        self.visit_expr(node.left)
        self.visit_expr(node.right)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.ensure(
            isinstance(node.op, (ast.And, ast.Or)), node, "Only and/or are supported"
        )
        for v in node.values:
            self.visit_expr(v)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.ensure(
            len(node.ops) == 1 and len(node.comparators) == 1,
            node,
            "Only simple binary comparisons are supported",
        )
        self.ensure(
            isinstance(
                node.ops[0], (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
            ),
            node,
            "Only ==, !=, <, <=, >, >= are supported",
        )
        self.visit_expr(node.left)
        self.visit_expr(node.comparators[0])

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit_expr(node.test)
        self.visit_expr(node.body)
        self.visit_expr(node.orelse)
