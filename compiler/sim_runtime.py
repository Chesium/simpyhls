import ast
import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from .ast_checker import DSLFrontendChecker
from .ast_to_hir import lower_source
from .hir_to_lir import lower_func


class SimulationError(Exception):
    pass


@dataclass
class TraceEntry:
    kind: str
    name: str
    ports: dict[str, object]
    result: object
    latency: int
    step: Optional[int] = None
    cycle: Optional[int] = None


@dataclass
class PrimitiveModel:
    name: str
    impl: Callable[..., object]
    kind: Optional[str] = None
    latency: Optional[int] = None
    ports: Optional[Iterable[str]] = None

    def __post_init__(self) -> None:
        if self.kind is None:
            self.kind = "comb" if self.name.endswith("_comb") else "blocking"
        if self.kind not in {"comb", "blocking"}:
            raise SimulationError(f"Primitive '{self.name}' has invalid kind '{self.kind}'")

        if self.latency is None:
            self.latency = 0 if self.kind == "comb" else 1
        if self.latency < 0:
            raise SimulationError(f"Primitive '{self.name}' cannot use negative latency")

        if self.kind == "comb" and self.latency != 0:
            raise SimulationError(
                f"Combinational primitive '{self.name}' must use latency 0, got {self.latency}"
            )

        if self.ports is not None:
            self.ports = tuple(self.ports)

    def expected_ports(self) -> tuple[str, ...]:
        if self.ports is not None:
            return tuple(self.ports)

        signature = inspect.signature(self.impl)
        params = list(signature.parameters.values())
        if not params:
            raise SimulationError(
                f"Primitive '{self.name}' must accept tb as its first argument"
            )

        inferred: list[str] = []
        for param in params[1:]:
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise SimulationError(
                    f"Primitive '{self.name}' cannot infer ports from *args; declare ports explicitly"
                )
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise SimulationError(
                    f"Primitive '{self.name}' cannot infer ports from **kwargs; declare ports explicitly"
                )
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                raise SimulationError(
                    f"Primitive '{self.name}' port '{param.name}' must accept keyword passing"
                )
            inferred.append(param.name)
        return tuple(inferred)

    def validate_ports(self, ports: dict[str, object]) -> None:
        expected = self.expected_ports()
        missing = [name for name in expected if name not in ports]
        extra = [name for name in ports if name not in expected]
        if missing or extra:
            raise SimulationError(
                f"Primitive '{self.name}' port mismatch: missing={missing}, extra={extra}"
            )


@dataclass
class SimulationHarness:
    params: dict[str, object]
    primitives: dict[str, PrimitiveModel] | Iterable[PrimitiveModel]
    initial_state: object
    trace_debug: bool = False

    def __post_init__(self) -> None:
        self.params = dict(self.params)
        if isinstance(self.primitives, dict):
            normalized = dict(self.primitives)
        else:
            normalized = {primitive.name: primitive for primitive in self.primitives}

        for name, primitive in normalized.items():
            if not isinstance(primitive, PrimitiveModel):
                raise SimulationError(
                    f"Harness primitive '{name}' must be a PrimitiveModel instance"
                )
            if name != primitive.name:
                raise SimulationError(
                    f"Harness primitive key '{name}' does not match model name '{primitive.name}'"
                )
        self.primitives = normalized


@dataclass
class SimulationResult:
    return_value: object
    locals: dict[str, object]
    final_state: object
    trace: list[TraceEntry]
    stats: dict[str, int]


@dataclass
class EquivalenceReport:
    python_result: SimulationResult
    hir_result: SimulationResult
    lir_result: SimulationResult
    ok: bool
    mismatches: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "OK" if self.ok else "MISMATCH"
        lines = [f"EquivalenceReport[{status}]"]
        if self.mismatches:
            lines.extend(f"  - {item}" for item in self.mismatches)
        return "\n".join(lines)


class _SimulationContext:
    def __init__(self, harness: SimulationHarness, *, mode: str):
        self.mode = mode
        self.params = dict(harness.params)
        self.primitives = dict(harness.primitives)
        self.state = copy.deepcopy(harness.initial_state)
        self.trace_debug = harness.trace_debug
        self.trace: list[TraceEntry] = []
        self.stats: dict[str, int] = {
            "primitive_calls": 0,
            "comb_calls": 0,
            "blocking_calls": 0,
            "latency_cycles": 0,
            "logical_steps": 0,
        }

    def bump_steps(self, count: int = 1) -> None:
        self.stats["logical_steps"] += count

    def call_primitive(
        self,
        name: str,
        ports: dict[str, object],
        *,
        step: Optional[int] = None,
        cycle: Optional[int] = None,
    ) -> object:
        try:
            primitive = self.primitives[name]
        except KeyError as exc:
            raise SimulationError(f"Unknown primitive '{name}' in {self.mode} simulation") from exc

        primitive.validate_ports(ports)
        result = primitive.impl(self, **ports)

        self.stats["primitive_calls"] += 1
        if primitive.kind == "comb":
            self.stats["comb_calls"] += 1
        else:
            self.stats["blocking_calls"] += 1
            self.stats["latency_cycles"] += primitive.latency

        trace_entry = TraceEntry(
            kind=primitive.kind,
            name=name,
            ports=copy.deepcopy(dict(ports)),
            result=copy.deepcopy(result),
            latency=primitive.latency,
            step=step if self.trace_debug else None,
            cycle=cycle if self.trace_debug else None,
        )
        self.trace.append(trace_entry)
        return result


def evaluate_expr(expr: object, env: dict[str, object], ctx: _SimulationContext) -> object:
    from .hir import BinOp, Call, Compare, Const, Select, UnaryOp, VarRef

    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, VarRef):
        return env[expr.var.name]
    if isinstance(expr, UnaryOp):
        value = evaluate_expr(expr.value, env, ctx)
        if expr.op == "+":
            return +value
        if expr.op == "-":
            return -value
        if expr.op == "not":
            return not value
        raise SimulationError(f"Unsupported unary op '{expr.op}'")
    if isinstance(expr, BinOp):
        if expr.op == "and":
            lhs = evaluate_expr(expr.lhs, env, ctx)
            return evaluate_expr(expr.rhs, env, ctx) if lhs else lhs
        if expr.op == "or":
            lhs = evaluate_expr(expr.lhs, env, ctx)
            return lhs if lhs else evaluate_expr(expr.rhs, env, ctx)

        lhs = evaluate_expr(expr.lhs, env, ctx)
        rhs = evaluate_expr(expr.rhs, env, ctx)
        if expr.op == "+":
            return lhs + rhs
        if expr.op == "-":
            return lhs - rhs
        if expr.op == "*":
            return lhs * rhs
        if expr.op == "&":
            return lhs & rhs
        if expr.op == "|":
            return lhs | rhs
        if expr.op == "^":
            return lhs ^ rhs
        if expr.op == "<<":
            return lhs << rhs
        if expr.op == ">>":
            return lhs >> rhs
        raise SimulationError(f"Unsupported binary op '{expr.op}'")
    if isinstance(expr, Compare):
        lhs = evaluate_expr(expr.lhs, env, ctx)
        rhs = evaluate_expr(expr.rhs, env, ctx)
        if expr.op == "==":
            return lhs == rhs
        if expr.op == "!=":
            return lhs != rhs
        if expr.op == "<":
            return lhs < rhs
        if expr.op == "<=":
            return lhs <= rhs
        if expr.op == ">":
            return lhs > rhs
        if expr.op == ">=":
            return lhs >= rhs
        raise SimulationError(f"Unsupported compare op '{expr.op}'")
    if isinstance(expr, Select):
        cond = evaluate_expr(expr.cond, env, ctx)
        branch = expr.true_value if cond else expr.false_value
        return evaluate_expr(branch, env, ctx)
    if isinstance(expr, Call):
        ports = {
            arg.name: evaluate_expr(arg.value, env, ctx)
            for arg in expr.args
        }
        return ctx.call_primitive(expr.func, ports)
    raise SimulationError(f"Unsupported expression type '{type(expr).__name__}'")


def snapshot_locals(local_names: Iterable[str], env: dict[str, object]) -> dict[str, object]:
    return {name: copy.deepcopy(env[name]) for name in local_names}


def _normalize_trace(trace: list[TraceEntry]) -> list[tuple[object, ...]]:
    normalized: list[tuple[object, ...]] = []
    for entry in trace:
        normalized.append(
            (
                entry.kind,
                entry.name,
                dict(entry.ports),
                entry.result,
                entry.latency,
            )
        )
    return normalized


def compare_simulation_results(
    baseline_name: str,
    baseline: SimulationResult,
    candidate_name: str,
    candidate: SimulationResult,
) -> list[str]:
    mismatches: list[str] = []

    if baseline.return_value != candidate.return_value:
        mismatches.append(
            f"{baseline_name} vs {candidate_name}: return_value differs "
            f"({baseline.return_value!r} != {candidate.return_value!r})"
        )

    if baseline.locals != candidate.locals:
        mismatches.append(
            f"{baseline_name} vs {candidate_name}: locals differ "
            f"({baseline.locals!r} != {candidate.locals!r})"
        )

    if baseline.final_state != candidate.final_state:
        mismatches.append(
            f"{baseline_name} vs {candidate_name}: final_state differs"
        )

    base_trace = _normalize_trace(baseline.trace)
    cand_trace = _normalize_trace(candidate.trace)
    if base_trace != cand_trace:
        mismatch_index = next(
            (
                index
                for index, (lhs, rhs) in enumerate(zip(base_trace, cand_trace))
                if lhs != rhs
            ),
            min(len(base_trace), len(cand_trace)),
        )
        baseline_entry = base_trace[mismatch_index] if mismatch_index < len(base_trace) else None
        candidate_entry = cand_trace[mismatch_index] if mismatch_index < len(cand_trace) else None
        mismatches.append(
            f"{baseline_name} vs {candidate_name}: trace differs at index {mismatch_index} "
            f"({baseline_entry!r} != {candidate_entry!r})"
        )

    return mismatches


def build_equivalence_report(
    python_result: SimulationResult,
    hir_result: SimulationResult,
    lir_result: SimulationResult,
) -> EquivalenceReport:
    mismatches = []
    mismatches.extend(compare_simulation_results("python", python_result, "hir", hir_result))
    mismatches.extend(compare_simulation_results("python", python_result, "lir", lir_result))
    return EquivalenceReport(
        python_result=python_result,
        hir_result=hir_result,
        lir_result=lir_result,
        ok=not mismatches,
        mismatches=mismatches,
    )


class _LocalCaptureTransformer(ast.NodeTransformer):
    def __init__(self, function_name: str, locals_in_order: list[str]):
        self.function_name = function_name
        self.locals_in_order = locals_in_order
        self.locals_set = set(locals_in_order)
        self._active_function_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name != self.function_name:
            return node

        self._active_function_depth += 1
        try:
            node = self.generic_visit(node)
            node.body.extend(self._snapshot_all())
            return node
        finally:
            self._active_function_depth -= 1

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        if self._active_function_depth != 1:
            return node
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        if node.targets[0].id not in self.locals_set:
            return node
        return [node, self._sync_local(node.targets[0].id)]

    def visit_For(self, node: ast.For):
        node = self.generic_visit(node)
        if (
            self._active_function_depth == 1
            and isinstance(node.target, ast.Name)
            and node.target.id in self.locals_set
        ):
            node.body = [self._sync_local(node.target.id), *node.body]
        return node

    def visit_Return(self, node: ast.Return):
        node = self.generic_visit(node)
        if self._active_function_depth != 1:
            return node
        return [*self._snapshot_all(), node]

    def _sync_local(self, name: str) -> ast.Assign:
        return ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id="__simpyhls_locals__", ctx=ast.Load()),
                    slice=ast.Constant(value=name),
                    ctx=ast.Store(),
                )
            ],
            value=ast.Name(id=name, ctx=ast.Load()),
        )

    def _snapshot_all(self) -> list[ast.Assign]:
        return [self._sync_local(name) for name in self.locals_in_order]


def run_python(source: str, harness: SimulationHarness) -> SimulationResult:
    checker = DSLFrontendChecker()
    frontend_info = checker.check_source(source)

    missing_params = [name for name in frontend_info.params if name not in harness.params]
    extra_params = [name for name in harness.params if name not in frontend_info.params]
    if missing_params or extra_params:
        raise SimulationError(
            f"Parameter mismatch: missing={missing_params}, extra={extra_params}"
        )

    missing_primitives = [
        name for name in frontend_info.primitives if name not in harness.primitives
    ]
    if missing_primitives:
        raise SimulationError(
            f"Missing primitive models for: {missing_primitives}"
        )

    tree = ast.parse(source)
    tree = _LocalCaptureTransformer(
        frontend_info.function_name,
        list(frontend_info.locals.keys()),
    ).visit(tree)
    ast.fix_missing_locations(tree)

    ctx = _SimulationContext(harness, mode="python")
    locals_capture: dict[str, object] = {}

    exec_globals = {
        "__builtins__": __builtins__,
        "__simpyhls_locals__": locals_capture,
    }

    def make_wrapper(name: str):
        def wrapper(**ports):
            return ctx.call_primitive(name, ports)

        return wrapper

    for primitive_name in frontend_info.primitives:
        exec_globals[primitive_name] = make_wrapper(primitive_name)

    exec(compile(tree, "<simpyhls-dsl>", "exec"), exec_globals, exec_globals)
    func = exec_globals[frontend_info.function_name]
    args = [harness.params[name] for name in frontend_info.params]
    return_value = func(*args)

    result_locals = {
        name: copy.deepcopy(locals_capture[name])
        for name in frontend_info.locals
    }
    return SimulationResult(
        return_value=copy.deepcopy(return_value),
        locals=result_locals,
        final_state=ctx.state,
        trace=ctx.trace,
        stats=copy.deepcopy(ctx.stats),
    )


def run_hir(func_ir, harness: SimulationHarness) -> SimulationResult:
    from .hir_sim import simulate_func

    return simulate_func(func_ir, harness)


def run_lir(func_lir, harness: SimulationHarness) -> SimulationResult:
    from .lir_sim import simulate_func

    return simulate_func(func_lir, harness)


def run_all(source: str, harness: SimulationHarness) -> EquivalenceReport:
    func_ir = lower_source(source)
    func_lir = lower_func(func_ir)
    python_result = run_python(source, harness)
    hir_result = run_hir(func_ir, harness)
    lir_result = run_lir(func_lir, harness)
    return build_equivalence_report(python_result, hir_result, lir_result)


__all__ = [
    "EquivalenceReport",
    "PrimitiveModel",
    "SimulationError",
    "SimulationHarness",
    "SimulationResult",
    "TraceEntry",
    "build_equivalence_report",
    "compare_simulation_results",
    "evaluate_expr",
    "run_all",
    "run_hir",
    "run_lir",
    "run_python",
    "snapshot_locals",
]
