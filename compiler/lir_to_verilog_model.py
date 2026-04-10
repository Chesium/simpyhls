import math
import re
import struct
from dataclasses import dataclass, field
from typing import Optional

from .hir import BinOp, Call, Compare, Const, Expr, Select, Type, UnaryOp, VarRef
from .lir import AssignOp, Await, Branch, FuncLIR, Jump, Return, SideEffectOp, StartOp
from .primitive_rtl import PrimitiveRTLRegistry, PrimitiveRTLSpec


class VerilogModelError(Exception):
    pass


@dataclass
class RTLModuleConfig:
    module_name: Optional[str] = None
    clock_signal: str = "clk"
    reset_signal: str = "rst_n"
    start_signal: str = "start"
    busy_signal: str = "busy"
    done_signal: str = "done"
    return_signal: str = "ret_val"
    emit_comments: bool = True
    package_imports: tuple[str, ...] = ()


@dataclass
class RTLComment:
    lines: list[str] = field(default_factory=list)


@dataclass
class RTLPort:
    direction: str
    name: str
    width: int = 1
    comment: Optional[str] = None


@dataclass
class RTLReg:
    name: str
    width: int = 1
    comment: Optional[str] = None


@dataclass
class RTLAssign:
    target: str
    expr: str


@dataclass
class RTLPrimitiveStart:
    spec_name: str
    start_signal: str
    arg_assigns: list[RTLAssign] = field(default_factory=list)
    result_target: Optional[str] = None
    result_signal: Optional[str] = None


@dataclass
class RTLJumpTransition:
    kind: str = field(init=False, default="jump")
    target: str = ""


@dataclass
class RTLBranchTransition:
    cond: str
    true_target: str
    false_target: str
    kind: str = field(init=False, default="branch")


@dataclass
class RTLWaitTransition:
    done_signal: str
    target: str
    capture_assigns: list[RTLAssign] = field(default_factory=list)
    kind: str = field(init=False, default="wait")


@dataclass
class RTLReturnTransition:
    value_expr: Optional[str] = None
    kind: str = field(init=False, default="return")


@dataclass
class RTLState:
    name: str
    lir_label: str
    comment: RTLComment
    next_assigns: list[RTLAssign] = field(default_factory=list)
    output_assigns: list[RTLAssign] = field(default_factory=list)
    primitive_start: Optional[RTLPrimitiveStart] = None
    transition: object = None


@dataclass
class RTLModule:
    name: str
    header_comments: list[str]
    package_imports: list[str]
    ports: list[RTLPort]
    regs: list[RTLReg]
    temps: list[RTLReg]
    next_regs: list[RTLReg]
    state_width: int
    state_names: list[str]
    idle_state: str
    done_state: str
    entry_state: str
    states: list[RTLState]
    clock_signal: str
    reset_signal: str
    start_signal: str
    busy_signal: str
    done_signal: str
    return_reg: Optional[RTLReg]
    return_signal: Optional[str]
    debug: dict[str, object] = field(default_factory=dict)


def width_from_type(typ: Type) -> int:
    if typ.name == "bool":
        return 1
    match = re.search(r"(\d+)", typ.name)
    if match:
        return max(1, int(match.group(1)))
    return 32


def sanitize_identifier(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not text:
        return "unnamed"
    if text[0].isdigit():
        return f"n_{text}"
    return text


def _infer_expr_type(expr: Expr) -> Optional[Type]:
    if isinstance(expr, Const):
        return expr.typ
    if isinstance(expr, VarRef):
        return expr.var.typ
    if isinstance(expr, UnaryOp):
        return _infer_expr_type(expr.value)
    if isinstance(expr, BinOp):
        if expr.op in {"and", "or"}:
            return Type("bool")
        return _infer_expr_type(expr.lhs) or _infer_expr_type(expr.rhs)
    if isinstance(expr, Compare):
        return Type("bool")
    if isinstance(expr, Select):
        return _infer_expr_type(expr.true_value) or _infer_expr_type(expr.false_value)
    if isinstance(expr, Call) and expr.args:
        return _infer_expr_type(expr.args[0].value)
    return None


def render_verilog_expr(
    expr: Expr,
    registry: PrimitiveRTLRegistry,
    env: Optional[dict[str, str]] = None,
) -> str:
    env = env or {}
    if isinstance(expr, Const):
        if isinstance(expr.value, bool):
            return "1'b1" if expr.value else "1'b0"
        if expr.typ.name.startswith("f"):
            width = width_from_type(expr.typ)
            if width == 32:
                bits = struct.unpack(">I", struct.pack(">f", float(expr.value)))[0]
                return f"32'h{bits:08x}"
            if width == 64:
                bits = struct.unpack(">Q", struct.pack(">d", float(expr.value)))[0]
                return f"64'h{bits:016x}"
            raise VerilogModelError(
                f"Unsupported floating-point constant width {width} for value {expr.value!r}"
            )
        width = width_from_type(expr.typ)
        if isinstance(expr.value, int) and expr.value < 0:
            return f"-{width}'sd{abs(expr.value)}"
        return f"{width}'d{expr.value}"
    if isinstance(expr, VarRef):
        return env.get(expr.var.name, sanitize_identifier(expr.var.name))
    if isinstance(expr, UnaryOp):
        op = "!" if expr.op == "not" else expr.op
        return f"({op} {render_verilog_expr(expr.value, registry, env)})"
    if isinstance(expr, BinOp):
        lhs = render_verilog_expr(expr.lhs, registry, env)
        rhs = render_verilog_expr(expr.rhs, registry, env)
        op = "&&" if expr.op == "and" else "||" if expr.op == "or" else expr.op
        return f"({lhs} {op} {rhs})"
    if isinstance(expr, Compare):
        return (
            f"({render_verilog_expr(expr.lhs, registry, env)} {expr.op} "
            f"{render_verilog_expr(expr.rhs, registry, env)})"
        )
    if isinstance(expr, Select):
        return (
            f"({render_verilog_expr(expr.cond, registry, env)} ? "
            f"{render_verilog_expr(expr.true_value, registry, env)} : "
            f"{render_verilog_expr(expr.false_value, registry, env)})"
        )
    if isinstance(expr, Call):
        spec = registry.require(expr.func)
        if spec.kind != "comb":
            raise VerilogModelError(
                f"Blocking primitive '{expr.func}' cannot appear in a combinational expression"
            )
        args = ", ".join(render_verilog_expr(arg.value, registry, env) for arg in expr.args)
        return f"{sanitize_identifier(spec.comb_function or spec.name)}({args})"
    raise VerilogModelError(f"Unsupported expression type '{type(expr).__name__}' for Verilog rendering")


def _collect_blocking_specs(func_lir: FuncLIR, registry: PrimitiveRTLRegistry) -> list[PrimitiveRTLSpec]:
    names: list[str] = []
    for block in func_lir.blocks.values():
        for op in block.ops:
            if isinstance(op, StartOp):
                spec = registry.require(op.call.func)
                if spec.kind != "blocking":
                    raise VerilogModelError(
                        f"StartOp uses non-blocking primitive '{op.call.func}'"
                    )
                names.append(op.call.func)
    return registry.used(names)


def _collect_blocking_port_widths(
    func_lir: FuncLIR,
    registry: PrimitiveRTLRegistry,
) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    arg_widths: dict[tuple[str, str], int] = {}
    result_widths: dict[str, int] = {}
    for block in func_lir.blocks.values():
        for op in block.ops:
            if not isinstance(op, StartOp):
                continue
            spec = registry.require(op.call.func)
            if spec.kind != "blocking":
                continue
            for arg in op.call.args:
                expr_type = _infer_expr_type(arg.value)
                if expr_type is None:
                    continue
                arg_widths.setdefault((spec.name, arg.name), width_from_type(expr_type))
            if op.result is not None:
                result_widths.setdefault(spec.name, width_from_type(op.result.typ))
    return arg_widths, result_widths


def _state_name(label: str) -> str:
    return f"S_{sanitize_identifier(label).upper()}"


def _wait_state_name(label: str) -> str:
    return f"{_state_name(label)}_WAIT"


def _comment_lines(label: str, block_comment: Optional[str], emit_comments: bool) -> RTLComment:
    if not emit_comments:
        return RTLComment()
    lines = [f"LIR block: {label}"]
    if block_comment:
        lines.append(block_comment)
    return RTLComment(lines)


def lower_to_verilog_model(
    func_lir: FuncLIR,
    primitive_registry: PrimitiveRTLRegistry,
    config: Optional[RTLModuleConfig] = None,
) -> RTLModule:
    config = config or RTLModuleConfig()
    blocking_specs = _collect_blocking_specs(func_lir, primitive_registry)
    arg_widths, result_widths = _collect_blocking_port_widths(func_lir, primitive_registry)

    state_lookup = {label: _state_name(label) for label in func_lir.blocks}
    wait_state_lookup = {
        label: _wait_state_name(label)
        for label, block in func_lir.blocks.items()
        if isinstance(block.term, Await)
    }
    idle_state = "S_IDLE"
    done_state = "S_DONE"
    entry_state = state_lookup[func_lir.entry]
    state_names = [idle_state]
    for label in func_lir.blocks:
        state_names.append(state_lookup[label])
        if label in wait_state_lookup:
            state_names.append(wait_state_lookup[label])
    state_names.append(done_state)
    state_width = max(1, math.ceil(math.log2(len(state_names))))

    ports = [
        RTLPort("input", sanitize_identifier(config.clock_signal)),
        RTLPort("input", sanitize_identifier(config.reset_signal)),
        RTLPort("input", sanitize_identifier(config.start_signal)),
        RTLPort("output", sanitize_identifier(config.busy_signal)),
        RTLPort("output", sanitize_identifier(config.done_signal)),
    ]
    ports.extend(
        RTLPort("input", sanitize_identifier(arg.name), width_from_type(arg.typ))
        for arg in func_lir.args
    )

    return_reg: Optional[RTLReg] = None
    return_signal: Optional[str] = None
    has_return_value = any(
        isinstance(block.term, Return) and block.term.value is not None
        for block in func_lir.blocks.values()
    )
    if has_return_value:
        return_type = None
        for block in func_lir.blocks.values():
            if isinstance(block.term, Return) and block.term.value is not None:
                return_type = _infer_expr_type(block.term.value)
                if return_type is not None:
                    break
        width = width_from_type(return_type or Type("i32"))
        return_signal = sanitize_identifier(config.return_signal)
        ports.append(RTLPort("output", return_signal, width))
        return_reg = RTLReg(return_signal, width, comment="latched function return value")

    for spec in blocking_specs:
        ports.append(RTLPort("output", sanitize_identifier(spec.start_signal)))
        for port_name in spec.ports:
            width = arg_widths.get((spec.name, port_name), 32)
            ports.append(
                RTLPort(
                    "output",
                    sanitize_identifier(spec.arg_signals[port_name]),
                    width,
                )
            )
        ports.append(RTLPort("input", sanitize_identifier(spec.done_signal)))
        if spec.result_signal and spec.name in result_widths:
            ports.append(
                RTLPort(
                    "input",
                    sanitize_identifier(spec.result_signal),
                    result_widths[spec.name],
                )
            )

    regs = [
        RTLReg(sanitize_identifier(var.name), width_from_type(var.typ), comment=f"local {var.typ.name}")
        for var in func_lir.locals
    ]
    temps = [
        RTLReg(sanitize_identifier(var.name), width_from_type(var.typ), comment="internal temporary")
        for var in func_lir.temps
    ]
    next_regs = [
        RTLReg(f"next_{reg.name}", reg.width, comment=f"next value for {reg.name}")
        for reg in [*regs, *temps, *([return_reg] if return_reg else [])]
    ]

    states: list[RTLState] = [
        RTLState(
            name=idle_state,
            lir_label="__idle__",
            comment=RTLComment(["FSM idle state", "Wait for start to enter the LIR entry block"])
            if config.emit_comments
            else RTLComment(),
            transition=RTLBranchTransition(
                cond=sanitize_identifier(config.start_signal),
                true_target=entry_state,
                false_target=idle_state,
            ),
        )
    ]

    for label, block in func_lir.blocks.items():
        state = RTLState(
            name=state_lookup[label],
            lir_label=label,
            comment=_comment_lines(label, block.comment, config.emit_comments),
        )

        pending_start: Optional[tuple[PrimitiveRTLSpec, StartOp]] = None
        expr_env: dict[str, str] = {}
        for op in block.ops:
            if isinstance(op, AssignOp):
                rendered_expr = render_verilog_expr(op.value, primitive_registry, expr_env)
                state.next_assigns.append(
                    RTLAssign(
                        target=sanitize_identifier(op.target.name),
                        expr=rendered_expr,
                    )
                )
                expr_env[op.target.name] = rendered_expr
                continue

            if isinstance(op, SideEffectOp):
                state.comment.lines.append(
                    f"comb expr: {render_verilog_expr(op.call, primitive_registry, expr_env)}"
                )
                continue

            if isinstance(op, StartOp):
                spec = primitive_registry.require(op.call.func)
                if spec.kind != "blocking":
                    raise VerilogModelError(
                        f"StartOp expects blocking primitive, got '{spec.name}'"
                    )
                arg_assigns = [
                    RTLAssign(
                        target=sanitize_identifier(spec.arg_signals[arg.name]),
                        expr=render_verilog_expr(arg.value, primitive_registry, expr_env),
                    )
                    for arg in op.call.args
                ]
                state.output_assigns.extend(arg_assigns)
                state.output_assigns.append(
                    RTLAssign(
                        target=sanitize_identifier(spec.start_signal),
                        expr="1'b1",
                    )
                )
                state.primitive_start = RTLPrimitiveStart(
                    spec_name=spec.name,
                    start_signal=sanitize_identifier(spec.start_signal),
                    arg_assigns=arg_assigns,
                    result_target=sanitize_identifier(op.result.name) if op.result is not None else None,
                    result_signal=sanitize_identifier(spec.result_signal) if spec.result_signal else None,
                )
                pending_start = (spec, op)
                continue

            raise VerilogModelError(f"Unsupported LIR op '{type(op).__name__}'")

        term = block.term
        if isinstance(term, Jump):
            state.transition = RTLJumpTransition(target=state_lookup[term.target])
        elif isinstance(term, Branch):
            state.transition = RTLBranchTransition(
                cond=render_verilog_expr(term.cond, primitive_registry, expr_env),
                true_target=state_lookup[term.true_target],
                false_target=state_lookup[term.false_target],
            )
        elif isinstance(term, Await):
            if pending_start is None:
                raise VerilogModelError(
                    f"LIR block '{label}' awaits a primitive without a preceding StartOp"
                )
            spec, start_op = pending_start
            wait_state_name = wait_state_lookup[label]
            state.transition = RTLJumpTransition(target=wait_state_name)

            capture_assigns: list[RTLAssign] = []
            if start_op.result is not None:
                if not spec.result_signal:
                    raise VerilogModelError(
                        f"Blocking primitive '{spec.name}' captures a result but has no result_signal"
                    )
                capture_assigns.append(
                    RTLAssign(
                        target=sanitize_identifier(start_op.result.name),
                        expr=sanitize_identifier(spec.result_signal),
                    )
                )
            states.append(state)

            wait_comment = _comment_lines(label, block.comment, config.emit_comments)
            if config.emit_comments:
                wait_comment.lines.append(f"wait for blocking primitive: {spec.name}")
            wait_state = RTLState(
                name=wait_state_name,
                lir_label=f"{label}__wait",
                comment=wait_comment,
                transition=RTLWaitTransition(
                    done_signal=sanitize_identifier(spec.done_signal),
                    target=state_lookup[term.target],
                    capture_assigns=capture_assigns,
                ),
            )
            states.append(wait_state)
            continue
        elif isinstance(term, Return):
            state.transition = RTLReturnTransition(
                value_expr=render_verilog_expr(term.value, primitive_registry, expr_env)
                if term.value is not None
                else None
            )
        else:
            raise VerilogModelError(
                f"LIR block '{label}' has unsupported terminator '{type(term).__name__}'"
            )

        states.append(state)

    states.append(
        RTLState(
            name=done_state,
            lir_label="__done__",
            comment=RTLComment(["FSM done state", "Pulse done until start deasserts"])
            if config.emit_comments
            else RTLComment(),
            transition=RTLBranchTransition(
                cond=f"!{sanitize_identifier(config.start_signal)}",
                true_target=idle_state,
                false_target=done_state,
            ),
        )
    )

    header_comments = [
        f"Generated from LIR for function {func_lir.name}",
        f"Entry block: {func_lir.entry}",
    ]
    if blocking_specs:
        header_comments.append(
            "Blocking primitives: "
            + ", ".join(
                f"{spec.name}(latency={spec.latency})"
                for spec in blocking_specs
            )
        )

    return RTLModule(
        name=sanitize_identifier(config.module_name or func_lir.name),
        header_comments=header_comments,
        package_imports=list(config.package_imports),
        ports=ports,
        regs=regs,
        temps=temps,
        next_regs=next_regs,
        state_width=state_width,
        state_names=state_names,
        idle_state=idle_state,
        done_state=done_state,
        entry_state=entry_state,
        states=states,
        clock_signal=sanitize_identifier(config.clock_signal),
        reset_signal=sanitize_identifier(config.reset_signal),
        start_signal=sanitize_identifier(config.start_signal),
        busy_signal=sanitize_identifier(config.busy_signal),
        done_signal=sanitize_identifier(config.done_signal),
        return_reg=return_reg,
        return_signal=return_signal,
        debug={
            "state_lookup": state_lookup,
            "wait_state_lookup": wait_state_lookup,
            "wait_state_count": sum(
                1 for state in states if getattr(state.transition, "kind", None) == "wait"
            ),
            "blocking_primitives": [spec.name for spec in blocking_specs],
        },
    )


__all__ = [
    "RTLAssign",
    "RTLBranchTransition",
    "RTLComment",
    "RTLJumpTransition",
    "RTLModule",
    "RTLModuleConfig",
    "RTLPort",
    "RTLPrimitiveStart",
    "RTLReg",
    "RTLReturnTransition",
    "RTLState",
    "RTLWaitTransition",
    "VerilogModelError",
    "lower_to_verilog_model",
    "render_verilog_expr",
    "sanitize_identifier",
    "width_from_type",
]
