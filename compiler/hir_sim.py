import copy

from .hir import Assign, ExprStmt, ForRangeStmt, FuncIR, IfStmt, ReturnStmt, WhileStmt
from .sim_runtime import SimulationHarness, SimulationResult, _SimulationContext, evaluate_expr, snapshot_locals


class _FunctionReturn(Exception):
    def __init__(self, value: object):
        super().__init__()
        self.value = value


class HIRSimulator:
    def __init__(self, harness: SimulationHarness):
        self.harness = harness
        self.ctx = _SimulationContext(harness, mode="hir")
        self.env: dict[str, object] = dict(harness.params)

    def simulate_func(self, func_ir: FuncIR) -> SimulationResult:
        try:
            self._exec_stmts(func_ir.body)
            return_value = None
        except _FunctionReturn as returned:
            return_value = returned.value

        return SimulationResult(
            return_value=copy.deepcopy(return_value),
            locals=snapshot_locals([var.name for var in func_ir.locals], self.env),
            final_state=self.ctx.state,
            trace=self.ctx.trace,
            stats=copy.deepcopy(self.ctx.stats),
        )

    def _exec_stmts(self, stmts) -> None:
        for stmt in stmts:
            self._exec_stmt(stmt)

    def _exec_stmt(self, stmt) -> None:
        self.ctx.bump_steps()

        if isinstance(stmt, Assign):
            self.env[stmt.target.name] = evaluate_expr(stmt.value, self.env, self.ctx)
            return

        if isinstance(stmt, ExprStmt):
            evaluate_expr(stmt.value, self.env, self.ctx)
            return

        if isinstance(stmt, IfStmt):
            cond = evaluate_expr(stmt.cond, self.env, self.ctx)
            branch = stmt.then_body if cond else stmt.else_body
            self._exec_stmts(branch)
            return

        if isinstance(stmt, ForRangeStmt):
            start = evaluate_expr(stmt.start, self.env, self.ctx)
            stop = evaluate_expr(stmt.stop, self.env, self.ctx)
            for index in range(start, stop):
                self.env[stmt.iter_var.name] = index
                self._exec_stmts(stmt.body)
            return

        if isinstance(stmt, WhileStmt):
            while evaluate_expr(stmt.cond, self.env, self.ctx):
                self._exec_stmts(stmt.body)
            return

        if isinstance(stmt, ReturnStmt):
            value = evaluate_expr(stmt.value, self.env, self.ctx) if stmt.value is not None else None
            raise _FunctionReturn(value)

        raise TypeError(f"Unsupported HIR statement type '{type(stmt).__name__}'")


def simulate_func(func_ir: FuncIR, harness: SimulationHarness) -> SimulationResult:
    return HIRSimulator(harness).simulate_func(func_ir)


__all__ = ["HIRSimulator", "simulate_func"]
