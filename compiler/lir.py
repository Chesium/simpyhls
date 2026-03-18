from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hir import Call, Expr, SourceInfo, Type, Var, format_expr


BlockId = str


@dataclass
class ValueRef:
    """
    A named transient produced during lowering/scheduling.

    LIR can still reference HIR expressions directly, but explicit temporary names
    are useful once primitive results or scheduled intermediate values need stable
    identities across blocks.
    """

    name: str
    typ: Optional[Type] = None


@dataclass
class Op:
    pass


@dataclass
class AssignOp(Op):
    """
    Single-step register/local update from a pure expression.

    Intended for constants, variable references, arithmetic, compares, selects,
    and combinational primitive calls.
    """

    target: Var
    value: Expr
    source_info: Optional[SourceInfo] = None


@dataclass
class EvalOp(Op):
    """
    Evaluate a pure expression into a temporary.

    Useful if later lowering wants explicit temporaries before branches or before
    assembling primitive arguments.
    """

    target: ValueRef
    value: Expr
    source_info: Optional[SourceInfo] = None


@dataclass
class StartOp(Op):
    """
    Launch a blocking primitive operation.

    The corresponding Await terminator models the control stall until completion.
    If the primitive returns a value, the lowerer should name the destination in
    result so later steps can capture or forward it explicitly.
    """

    call: Call
    result: Optional[Var] = None
    token: Optional[str] = None
    source_info: Optional[SourceInfo] = None


@dataclass
class CaptureOp(Op):
    """
    Materialize a completed primitive result into a variable.

    This is primarily useful when a backend chooses to separate "operation has
    completed" from "latched returned data".
    """

    target: Var
    source: ValueRef
    source_info: Optional[SourceInfo] = None


@dataclass
class SideEffectOp(Op):
    """
    Fire-and-forget side effect in the current step.

    Useful for stores or other operations that do not return a value and do not
    need a blocking handshake in the LIR.
    """

    call: Call
    source_info: Optional[SourceInfo] = None


@dataclass
class Terminator:
    pass


@dataclass
class Jump(Terminator):
    target: BlockId
    source_info: Optional[SourceInfo] = None


@dataclass
class Branch(Terminator):
    cond: Expr
    true_target: BlockId
    false_target: BlockId
    source_info: Optional[SourceInfo] = None


@dataclass
class Await(Terminator):
    """
    Stall until the most recently-started blocking primitive completes.

    In a later backend this usually maps to a state that keeps `busy=1`, drives
    request signals, and transitions to `target` when the primitive done/valid
    condition is observed.
    """

    target: BlockId
    token: Optional[str] = None
    source_info: Optional[SourceInfo] = None


@dataclass
class Return(Terminator):
    value: Optional[Expr] = None
    source_info: Optional[SourceInfo] = None


@dataclass
class BasicBlock:
    label: BlockId
    ops: List[Op] = field(default_factory=list)
    term: Optional[Terminator] = None
    comment: Optional[str] = None


@dataclass
class FuncLIR:
    name: str
    args: List[Var]
    locals: List[Var]
    entry: BlockId
    blocks: Dict[BlockId, BasicBlock]
    temps: List[Var] = field(default_factory=list)

    def __repr__(self) -> str:
        return format_func_lir(self)


def format_lir_op(op: Op, indent: int = 0) -> str:
    prefix = "    " * indent
    if isinstance(op, AssignOp):
        return f"{prefix}{op.target.name} <- {format_expr(op.value)}"
    if isinstance(op, EvalOp):
        return f"{prefix}{op.target.name} := {format_expr(op.value)}"
    if isinstance(op, StartOp):
        result = f"{op.result.name} = " if op.result is not None else ""
        token = f" [{op.token}]" if op.token else ""
        return f"{prefix}start {result}{op.call}{token}"
    if isinstance(op, CaptureOp):
        return f"{prefix}{op.target.name} <- {op.source.name}"
    if isinstance(op, SideEffectOp):
        return f"{prefix}do {op.call}"
    return f"{prefix}{op!r}"


def format_lir_term(term: Terminator, indent: int = 0) -> str:
    prefix = "    " * indent
    if isinstance(term, Jump):
        return f"{prefix}jump {term.target}"
    if isinstance(term, Branch):
        return (
            f"{prefix}branch {format_expr(term.cond)} "
            f"? {term.true_target} : {term.false_target}"
        )
    if isinstance(term, Await):
        token = f" [{term.token}]" if term.token else ""
        return f"{prefix}await -> {term.target}{token}"
    if isinstance(term, Return):
        if term.value is None:
            return f"{prefix}return"
        return f"{prefix}return {format_expr(term.value)}"
    return f"{prefix}{term!r}"


def format_block(block: BasicBlock) -> str:
    lines = [f"{block.label}:"]
    if block.comment:
        lines.append(f"    # {block.comment}")
    if block.ops:
        lines.extend(format_lir_op(op, indent=1) for op in block.ops)
    elif block.term is None:
        lines.append("    pass")
    if block.term is not None:
        lines.append(format_lir_term(block.term, indent=1))
    return "\n".join(lines)


def format_func_lir(func: FuncLIR) -> str:
    arg_names = ", ".join(var.name for var in func.args)
    lines = [f"lir func {func.name}({arg_names})", f"entry: {func.entry}"]
    if func.locals:
        lines.append("locals:")
        lines.extend(f"    {var.typ.name} {var.name}" for var in func.locals)
    lines.append("blocks:")
    for block in func.blocks.values():
        for line in format_block(block).splitlines():
            lines.append(f"    {line}")
    return "\n".join(lines)
