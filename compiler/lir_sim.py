import copy
from dataclasses import dataclass

from .lir import AssignOp, Await, Branch, FuncLIR, Jump, Return, SideEffectOp, StartOp
from .sim_runtime import SimulationError, SimulationHarness, SimulationResult, _SimulationContext, evaluate_expr, snapshot_locals


@dataclass
class _PendingInvocation:
    call: object
    result_name: str | None
    token: str | None


class LIRSimulator:
    def __init__(self, harness: SimulationHarness):
        self.harness = harness
        self.ctx = _SimulationContext(harness, mode="lir")
        self.env: dict[str, object] = dict(harness.params)
        self.pending: _PendingInvocation | None = None

    def simulate_func(self, func_lir: FuncLIR) -> SimulationResult:
        current = func_lir.entry
        return_value = None

        while True:
            block = func_lir.blocks[current]

            for op in block.ops:
                self.ctx.bump_steps()
                if isinstance(op, AssignOp):
                    self.env[op.target.name] = evaluate_expr(op.value, self.env, self.ctx)
                    continue

                if isinstance(op, SideEffectOp):
                    ports = {
                        arg.name: evaluate_expr(arg.value, self.env, self.ctx)
                        for arg in op.call.args
                    }
                    self.ctx.call_primitive(op.call.func, ports)
                    continue

                if isinstance(op, StartOp):
                    if self.pending is not None:
                        raise SimulationError("Cannot issue StartOp while another primitive is pending")
                    self.pending = _PendingInvocation(
                        call=op.call,
                        result_name=op.result.name if op.result is not None else None,
                        token=op.token,
                    )
                    continue

                raise SimulationError(f"Unsupported LIR op '{type(op).__name__}'")

            term = block.term
            if term is None:
                raise SimulationError(f"LIR block '{block.label}' is missing a terminator")

            self.ctx.bump_steps()

            if isinstance(term, Jump):
                current = term.target
                continue

            if isinstance(term, Branch):
                cond = evaluate_expr(term.cond, self.env, self.ctx)
                current = term.true_target if cond else term.false_target
                continue

            if isinstance(term, Await):
                if self.pending is None:
                    raise SimulationError("Await encountered with no pending primitive")
                if term.token != self.pending.token:
                    raise SimulationError(
                        f"Await token mismatch: expected {self.pending.token!r}, got {term.token!r}"
                    )

                ports = {
                    arg.name: evaluate_expr(arg.value, self.env, self.ctx)
                    for arg in self.pending.call.args
                }
                result = self.ctx.call_primitive(
                    self.pending.call.func,
                    ports,
                    cycle=self.ctx.stats["latency_cycles"],
                )
                if self.pending.result_name is not None:
                    self.env[self.pending.result_name] = result
                self.pending = None
                current = term.target
                continue

            if isinstance(term, Return):
                if self.pending is not None:
                    raise SimulationError("Return encountered while a primitive is still pending")
                return_value = evaluate_expr(term.value, self.env, self.ctx) if term.value is not None else None
                break

            raise SimulationError(f"Unsupported LIR terminator '{type(term).__name__}'")

        return SimulationResult(
            return_value=copy.deepcopy(return_value),
            locals=snapshot_locals([var.name for var in func_lir.locals], self.env),
            final_state=self.ctx.state,
            trace=self.ctx.trace,
            stats=copy.deepcopy(self.ctx.stats),
        )


def simulate_func(func_lir: FuncLIR, harness: SimulationHarness) -> SimulationResult:
    return LIRSimulator(harness).simulate_func(func_lir)


__all__ = ["LIRSimulator", "simulate_func"]
