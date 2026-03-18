from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SourceInfo:
    dsl_lineno: Optional[int] = None
    dsl_text: Optional[str] = None
    hir_note: Optional[str] = None
    lir_block: Optional[str] = None
    lir_op_index: Optional[int] = None


@dataclass
class Type:
    name: str  # "i32", "f32", "bool"


@dataclass
class Var:
    name: str
    typ: Type


@dataclass
class Expr:
    pass


@dataclass
class Const(Expr):
    value: object
    typ: Type


@dataclass
class VarRef(Expr):
    var: Var


@dataclass
class UnaryOp(Expr):
    op: str
    value: Expr


@dataclass
class BinOp(Expr):
    op: str
    lhs: Expr
    rhs: Expr


@dataclass
class Compare(Expr):
    op: str
    lhs: Expr
    rhs: Expr


@dataclass
class Select(Expr):
    cond: Expr
    true_value: Expr
    false_value: Expr


@dataclass
class CallArg:
    name: str
    value: Expr

    def __repr__(self) -> str:
        return f"{self.name}={format_expr(self.value)}"


@dataclass
class Call(Expr):
    func: str
    args: List[CallArg]

    def __repr__(self) -> str:
        return format_call(self)


@dataclass
class Stmt:
    pass


@dataclass
class Assign(Stmt):
    target: Var
    value: Expr
    source_info: Optional[SourceInfo] = None


@dataclass
class ExprStmt(Stmt):
    value: Expr
    source_info: Optional[SourceInfo] = None


@dataclass
class IfStmt(Stmt):
    cond: Expr
    then_body: List[Stmt]
    else_body: List[Stmt]
    source_info: Optional[SourceInfo] = None


@dataclass
class ForRangeStmt(Stmt):
    iter_var: Var
    start: Expr
    stop: Expr
    body: List[Stmt]
    source_info: Optional[SourceInfo] = None


@dataclass
class WhileStmt(Stmt):
    cond: Expr
    body: List[Stmt]
    source_info: Optional[SourceInfo] = None


@dataclass
class ReturnStmt(Stmt):
    value: Optional[Expr]
    source_info: Optional[SourceInfo] = None


@dataclass
class FuncIR:
    name: str
    args: List[Var]
    locals: List[Var]
    body: List[Stmt]
    source_info: Optional[SourceInfo] = None

    def __repr__(self) -> str:
        return format_func_ir(self)


def format_source_info(source_info: Optional[SourceInfo]) -> Optional[str]:
    if source_info is None:
        return None

    details: list[str] = []
    if source_info.dsl_lineno is not None:
        details.append(f"line {source_info.dsl_lineno}")
    if source_info.dsl_text:
        details.append(source_info.dsl_text.strip())
    elif source_info.hir_note:
        details.append(source_info.hir_note)
    return ": ".join(details) if details else None


def format_expr(expr: Expr) -> str:
    if isinstance(expr, Const):
        return repr(expr.value)
    if isinstance(expr, VarRef):
        return expr.var.name
    if isinstance(expr, UnaryOp):
        return f"({expr.op} {format_expr(expr.value)})"
    if isinstance(expr, BinOp):
        return f"({format_expr(expr.lhs)} {expr.op} {format_expr(expr.rhs)})"
    if isinstance(expr, Compare):
        return f"({format_expr(expr.lhs)} {expr.op} {format_expr(expr.rhs)})"
    if isinstance(expr, Select):
        return (
            f"({format_expr(expr.true_value)} if {format_expr(expr.cond)} "
            f"else {format_expr(expr.false_value)})"
        )
    if isinstance(expr, Call):
        return format_call(expr)
    return repr(expr)


def format_call(call: Call) -> str:
    args = ", ".join(repr(arg) for arg in call.args)
    return f"{call.func}({args})"


def format_stmt(stmt: Stmt, indent: int = 0) -> str:
    prefix = "    " * indent
    if isinstance(stmt, Assign):
        return f"{prefix}{stmt.target.name} = {format_expr(stmt.value)}"
    if isinstance(stmt, ExprStmt):
        return f"{prefix}{format_expr(stmt.value)}"
    if isinstance(stmt, IfStmt):
        lines = [f"{prefix}if {format_expr(stmt.cond)}:"]
        lines.extend(format_block(stmt.then_body, indent + 1))
        if stmt.else_body:
            lines.append(f"{prefix}else:")
            lines.extend(format_block(stmt.else_body, indent + 1))
        return "\n".join(lines)
    if isinstance(stmt, ForRangeStmt):
        header = (
            f"{prefix}for {stmt.iter_var.name} in "
            f"range({format_expr(stmt.start)}, {format_expr(stmt.stop)}):"
        )
        lines = [header]
        lines.extend(format_block(stmt.body, indent + 1))
        return "\n".join(lines)
    if isinstance(stmt, WhileStmt):
        lines = [f"{prefix}while {format_expr(stmt.cond)}:"]
        lines.extend(format_block(stmt.body, indent + 1))
        return "\n".join(lines)
    if isinstance(stmt, ReturnStmt):
        if stmt.value is None:
            return f"{prefix}return"
        return f"{prefix}return {format_expr(stmt.value)}"
    return f"{prefix}{stmt!r}"


def format_block(stmts: List[Stmt], indent: int = 0) -> List[str]:
    if not stmts:
        return [f"{'    ' * indent}pass"]
    return [line for stmt in stmts for line in format_stmt(stmt, indent).splitlines()]


def format_func_ir(func_ir: FuncIR) -> str:
    arg_names = ", ".join(var.name for var in func_ir.args)
    lines = [f"func {func_ir.name}({arg_names})"]
    if func_ir.locals:
        lines.append("locals:")
        lines.extend(f"    {var.typ.name} {var.name}" for var in func_ir.locals)
    lines.append("body:")
    lines.extend(format_block(func_ir.body, indent=1))
    return "\n".join(lines)
