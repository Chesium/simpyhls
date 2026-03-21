from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .ast_to_hir import lower_source
from .hir import (
    Assign,
    Call,
    Expr,
    ExprStmt,
    ForRangeStmt,
    FuncIR,
    IfStmt,
    ReturnStmt,
    WhileStmt,
    format_expr,
)
from .hir_to_lir import lower_func
from .lir import FuncLIR
from .lir_to_verilog_model import RTLModule, RTLModuleConfig, lower_to_verilog_model
from .primitive_rtl import PrimitiveRTLRegistry
from .verilog_codegen import render_verilog_module


@dataclass
class LatencyEstimate:
    lower_bound_cycles: int
    upper_bound_cycles: Optional[int]
    symbolic_expr: str
    exact: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PrimitiveUsageReport:
    name: str
    kind: str
    latency: int
    call_sites: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BlockingCallSiteReport:
    primitive: str
    latency: int
    line: Optional[int]
    text: Optional[str]
    context: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class CodegenReport:
    function_name: str
    module_name: str
    control_flow: dict[str, int]
    primitive_usage: list[PrimitiveUsageReport]
    blocking_call_sites: list[BlockingCallSiteReport]
    latency: LatencyEstimate
    lir_block_count: int
    rtl_state_count: int
    rtl_wait_state_count: int
    local_count: int
    temp_count: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "function_name": self.function_name,
            "module_name": self.module_name,
            "control_flow": dict(self.control_flow),
            "primitive_usage": [item.to_dict() for item in self.primitive_usage],
            "blocking_call_sites": [item.to_dict() for item in self.blocking_call_sites],
            "latency": self.latency.to_dict(),
            "lir_block_count": self.lir_block_count,
            "rtl_state_count": self.rtl_state_count,
            "rtl_wait_state_count": self.rtl_wait_state_count,
            "local_count": self.local_count,
            "temp_count": self.temp_count,
            "notes": list(self.notes),
        }


@dataclass
class CompilationArtifact:
    source: str
    hir: FuncIR
    lir: FuncLIR
    rtl_module: RTLModule
    verilog: str
    report: CodegenReport


def _find_call_expr(expr: Expr) -> Optional[Call]:
    return expr if isinstance(expr, Call) else None


def _tripcount_symbol(start: Expr, stop: Expr) -> str:
    if format_expr(start) == "0":
        return f"max({format_expr(stop)}, 0)"
    return f"max(({format_expr(stop)}) - ({format_expr(start)}), 0)"


def _sum_bounds(bounds: list[LatencyEstimate]) -> LatencyEstimate:
    lower = sum(item.lower_bound_cycles for item in bounds)
    upper = None if any(item.upper_bound_cycles is None for item in bounds) else sum(
        item.upper_bound_cycles or 0 for item in bounds
    )
    expr_parts = [item.symbolic_expr for item in bounds if item.symbolic_expr != "0"]
    symbolic_expr = " + ".join(expr_parts) if expr_parts else "0"
    notes: list[str] = []
    for item in bounds:
        notes.extend(item.notes)
    return LatencyEstimate(
        lower_bound_cycles=lower,
        upper_bound_cycles=upper,
        symbolic_expr=symbolic_expr,
        exact=all(item.exact for item in bounds),
        notes=notes,
    )


def _scale_bounds(bounds: LatencyEstimate, trip_count: int, trip_expr: str, *, exact: bool) -> LatencyEstimate:
    upper = None if bounds.upper_bound_cycles is None else trip_count * bounds.upper_bound_cycles
    symbolic_expr = "0"
    if bounds.symbolic_expr != "0":
        symbolic_expr = f"({trip_expr}) * ({bounds.symbolic_expr})"
    return LatencyEstimate(
        lower_bound_cycles=trip_count * bounds.lower_bound_cycles,
        upper_bound_cycles=upper,
        symbolic_expr=symbolic_expr,
        exact=exact and bounds.exact,
        notes=list(bounds.notes),
    )


def _stmt_latency(
    stmt,
    registry: PrimitiveRTLRegistry,
    context: list[str],
    blocking_sites: list[BlockingCallSiteReport],
    usage_counts: dict[str, int],
    control_flow: dict[str, int],
) -> LatencyEstimate:
    if isinstance(stmt, Assign):
        call = _find_call_expr(stmt.value)
        if call is None:
            return LatencyEstimate(0, 0, "0", True)
        spec = registry.require(call.func)
        usage_counts[spec.name] = usage_counts.get(spec.name, 0) + 1
        if spec.kind == "comb":
            return LatencyEstimate(0, 0, "0", True)
        blocking_sites.append(
            BlockingCallSiteReport(
                primitive=spec.name,
                latency=spec.latency,
                line=stmt.source_info.dsl_lineno if stmt.source_info else None,
                text=stmt.source_info.dsl_text.strip() if stmt.source_info and stmt.source_info.dsl_text else None,
                context=list(context),
            )
        )
        return LatencyEstimate(
            lower_bound_cycles=spec.latency,
            upper_bound_cycles=spec.latency,
            symbolic_expr=str(spec.latency),
            exact=True,
        )

    if isinstance(stmt, ExprStmt):
        call = _find_call_expr(stmt.value)
        if call is None:
            return LatencyEstimate(0, 0, "0", True)
        spec = registry.require(call.func)
        usage_counts[spec.name] = usage_counts.get(spec.name, 0) + 1
        if spec.kind == "comb":
            return LatencyEstimate(0, 0, "0", True)
        blocking_sites.append(
            BlockingCallSiteReport(
                primitive=spec.name,
                latency=spec.latency,
                line=stmt.source_info.dsl_lineno if stmt.source_info else None,
                text=stmt.source_info.dsl_text.strip() if stmt.source_info and stmt.source_info.dsl_text else None,
                context=list(context),
            )
        )
        return LatencyEstimate(
            lower_bound_cycles=spec.latency,
            upper_bound_cycles=spec.latency,
            symbolic_expr=str(spec.latency),
            exact=True,
        )

    if isinstance(stmt, IfStmt):
        control_flow["if"] += 1
        then_bounds = _sum_bounds(
            [
                _stmt_latency(child, registry, context + [f"if {format_expr(stmt.cond)}"], blocking_sites, usage_counts, control_flow)
                for child in stmt.then_body
            ]
        )
        else_bounds = _sum_bounds(
            [
                _stmt_latency(child, registry, context + [f"else of if {format_expr(stmt.cond)}"], blocking_sites, usage_counts, control_flow)
                for child in stmt.else_body
            ]
        )
        upper = None
        if then_bounds.upper_bound_cycles is not None and else_bounds.upper_bound_cycles is not None:
            upper = max(then_bounds.upper_bound_cycles, else_bounds.upper_bound_cycles)
        return LatencyEstimate(
            lower_bound_cycles=min(then_bounds.lower_bound_cycles, else_bounds.lower_bound_cycles),
            upper_bound_cycles=upper,
            symbolic_expr=(
                f"branch({format_expr(stmt.cond)}, {then_bounds.symbolic_expr}, {else_bounds.symbolic_expr})"
            ),
            exact=False,
            notes=then_bounds.notes + else_bounds.notes,
        )

    if isinstance(stmt, ForRangeStmt):
        control_flow["for"] += 1
        body_bounds = _sum_bounds(
            [
                _stmt_latency(
                    child,
                    registry,
                    context
                    + [f"for {stmt.iter_var.name} in range({format_expr(stmt.start)}, {format_expr(stmt.stop)})"],
                    blocking_sites,
                    usage_counts,
                    control_flow,
                )
                for child in stmt.body
            ]
        )
        trip_expr = _tripcount_symbol(stmt.start, stmt.stop)
        if format_expr(stmt.start).isdigit() and format_expr(stmt.stop).isdigit():
            start_value = int(format_expr(stmt.start))
            stop_value = int(format_expr(stmt.stop))
            trip_count = max(stop_value - start_value, 0)
            return _scale_bounds(body_bounds, trip_count, str(trip_count), exact=True)
        return LatencyEstimate(
            lower_bound_cycles=0,
            upper_bound_cycles=None,
            symbolic_expr=f"({trip_expr}) * ({body_bounds.symbolic_expr})",
            exact=False,
            notes=body_bounds.notes
            + [f"Loop latency depends on runtime trip count {trip_expr}"],
        )

    if isinstance(stmt, WhileStmt):
        control_flow["while"] += 1
        body_bounds = _sum_bounds(
            [
                _stmt_latency(
                    child,
                    registry,
                    context + [f"while {format_expr(stmt.cond)}"],
                    blocking_sites,
                    usage_counts,
                    control_flow,
                )
                for child in stmt.body
            ]
        )
        return LatencyEstimate(
            lower_bound_cycles=0,
            upper_bound_cycles=None,
            symbolic_expr=f"while({format_expr(stmt.cond)}) * ({body_bounds.symbolic_expr})",
            exact=False,
            notes=body_bounds.notes + ["While-loop latency is data dependent and has no static upper bound."],
        )

    if isinstance(stmt, ReturnStmt):
        return LatencyEstimate(0, 0, "0", True)

    return LatencyEstimate(0, None, "unknown", False, [f"Unsupported latency analysis node: {type(stmt).__name__}"])


def build_codegen_report(
    hir: FuncIR,
    lir: FuncLIR,
    rtl_module: RTLModule,
    primitive_registry: PrimitiveRTLRegistry,
) -> CodegenReport:
    usage_counts: dict[str, int] = {}
    blocking_sites: list[BlockingCallSiteReport] = []
    control_flow = {"if": 0, "for": 0, "while": 0}
    latency = _sum_bounds(
        [
            _stmt_latency(stmt, primitive_registry, [], blocking_sites, usage_counts, control_flow)
            for stmt in hir.body
        ]
    )

    primitive_usage = [
        PrimitiveUsageReport(
            name=spec.name,
            kind=spec.kind,
            latency=spec.latency,
            call_sites=usage_counts.get(spec.name, 0),
        )
        for spec in primitive_registry.specs.values()
        if usage_counts.get(spec.name, 0) > 0
    ]

    notes = [
        "Latency estimates count blocking primitive wait cycles only.",
        "Total end-to-end FSM cycles are larger because they also include control states and handshake transitions.",
    ]
    if latency.upper_bound_cycles is None:
        notes.append("Upper-bound latency is not statically known for data-dependent branches or loop trip counts.")

    return CodegenReport(
        function_name=hir.name,
        module_name=rtl_module.name,
        control_flow=control_flow,
        primitive_usage=primitive_usage,
        blocking_call_sites=blocking_sites,
        latency=latency,
        lir_block_count=len(lir.blocks),
        rtl_state_count=len(rtl_module.states),
        rtl_wait_state_count=rtl_module.debug.get("wait_state_count", 0),
        local_count=len(lir.locals),
        temp_count=len(lir.temps),
        notes=notes,
    )


def format_codegen_report(report: CodegenReport) -> str:
    lines = [
        f"Function: {report.function_name}",
        f"Module: {report.module_name}",
        f"LIR blocks: {report.lir_block_count}",
        f"RTL states: {report.rtl_state_count}",
        f"RTL wait states: {report.rtl_wait_state_count}",
        f"Locals: {report.local_count}",
        f"Temps: {report.temp_count}",
        (
            "Control flow: "
            f"if={report.control_flow['if']}, "
            f"for={report.control_flow['for']}, "
            f"while={report.control_flow['while']}"
        ),
        (
            "Blocking wait latency: "
            f"lower_bound={report.latency.lower_bound_cycles}, "
            f"upper_bound={report.latency.upper_bound_cycles if report.latency.upper_bound_cycles is not None else 'unknown'}"
        ),
        f"Blocking wait formula: {report.latency.symbolic_expr}",
    ]
    if report.primitive_usage:
        lines.append("Primitive usage:")
        lines.extend(
            f"  - {item.name}: kind={item.kind}, latency={item.latency}, call_sites={item.call_sites}"
            for item in report.primitive_usage
        )
    if report.blocking_call_sites:
        lines.append("Blocking call sites:")
        for site in report.blocking_call_sites:
            context = " > ".join(site.context) if site.context else "top-level"
            line_desc = f"line {site.line}" if site.line is not None else "line ?"
            text = f" :: {site.text}" if site.text else ""
            lines.append(
                f"  - {line_desc}: {site.primitive} (latency={site.latency}) [{context}]{text}"
            )
    if report.notes:
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in report.notes)
    return "\n".join(lines)


def compile_source(
    source: str,
    primitive_registry: PrimitiveRTLRegistry | dict | list,
    module_config: Optional[RTLModuleConfig] = None,
) -> CompilationArtifact:
    registry = (
        primitive_registry
        if isinstance(primitive_registry, PrimitiveRTLRegistry)
        else PrimitiveRTLRegistry(primitive_registry)
    )
    hir = lower_source(source)
    lir = lower_func(hir)
    rtl_module = lower_to_verilog_model(lir, registry, module_config)
    verilog = render_verilog_module(rtl_module)
    report = build_codegen_report(hir, lir, rtl_module, registry)
    return CompilationArtifact(
        source=source,
        hir=hir,
        lir=lir,
        rtl_module=rtl_module,
        verilog=verilog,
        report=report,
    )


def compile_file(
    path: str | Path,
    primitive_registry: PrimitiveRTLRegistry | dict | list,
    module_config: Optional[RTLModuleConfig] = None,
) -> CompilationArtifact:
    source_path = Path(path)
    return compile_source(source_path.read_text(), primitive_registry, module_config)


def write_compilation_outputs(
    artifact: CompilationArtifact,
    verilog_path: str | Path,
    report_path: Optional[str | Path] = None,
) -> None:
    verilog_file = Path(verilog_path)
    verilog_file.write_text(artifact.verilog)
    if report_path is not None:
        Path(report_path).write_text(format_codegen_report(artifact.report))


__all__ = [
    "BlockingCallSiteReport",
    "CodegenReport",
    "CompilationArtifact",
    "LatencyEstimate",
    "PrimitiveUsageReport",
    "build_codegen_report",
    "compile_file",
    "compile_source",
    "format_codegen_report",
    "write_compilation_outputs",
]
