from dataclasses import dataclass, replace
from typing import Optional

from .hir import Assign, BinOp, Call, Compare, Const, Expr, ExprStmt, ForRangeStmt, FuncIR, IfStmt, ReturnStmt, SourceInfo, Stmt, Type, Var, VarRef, WhileStmt, format_source_info
from .lir import AssignOp, Await, BasicBlock, Branch, FuncLIR, Jump, Return, SideEffectOp, StartOp


class HIRToLIRError(Exception):
    pass


@dataclass
class BlockBuilder:
    blocks: dict[str, BasicBlock]
    next_block_id: int = 0
    next_token_id: int = 0

    def make_block(self, prefix: str) -> str:
        label = f"{prefix}_{self.next_block_id}"
        self.next_block_id += 1
        self.blocks[label] = BasicBlock(label=label)
        return label

    def add_op(self, label: str, op: object) -> None:
        block = self.blocks[label]
        if block.term is not None:
            raise HIRToLIRError(f"Cannot append op to terminated block '{label}'")
        source_info = getattr(op, "source_info", None)
        if source_info is not None:
            setattr(
                op,
                "source_info",
                replace(source_info, lir_block=label, lir_op_index=len(block.ops)),
            )
            if block.comment is None:
                block.comment = format_source_info(getattr(op, "source_info"))
        block.ops.append(op)

    def terminate(self, label: str, term: object) -> None:
        block = self.blocks[label]
        if block.term is not None:
            raise HIRToLIRError(f"Block '{label}' already has terminator")
        source_info = getattr(term, "source_info", None)
        if source_info is not None:
            setattr(
                term,
                "source_info",
                replace(source_info, lir_block=label, lir_op_index=len(block.ops)),
            )
            if block.comment is None:
                block.comment = format_source_info(getattr(term, "source_info"))
        block.term = term

    def fresh_token(self) -> str:
        token = f"op{self.next_token_id}"
        self.next_token_id += 1
        return token


class HIRToLIRLowerer:
    def __init__(self) -> None:
        self.builder = BlockBuilder(blocks={"entry": BasicBlock(label="entry")})
        self.next_hidden_var_id = 0
        self.hidden_vars: list[Var] = []

    def lower_func(self, func_ir: FuncIR) -> FuncLIR:
        current = self._lower_stmts(func_ir.body, "entry")
        if current is not None and self.builder.blocks[current].term is None:
            self.builder.terminate(current, Return())
        return FuncLIR(
            name=func_ir.name,
            args=list(func_ir.args),
            locals=list(func_ir.locals),
            entry="entry",
            blocks=self.builder.blocks,
            temps=list(self.hidden_vars),
        )

    def _lower_stmts(self, stmts: list[Stmt], current: Optional[str]) -> Optional[str]:
        for stmt in stmts:
            if current is None:
                raise HIRToLIRError("Encountered unreachable HIR statements after terminator")
            current = self._lower_stmt(stmt, current)
        return current

    def _lower_stmt(self, stmt: Stmt, current: str) -> Optional[str]:
        if isinstance(stmt, Assign):
            return self._lower_assign(stmt, current)
        if isinstance(stmt, ExprStmt):
            return self._lower_expr_stmt(stmt, current)
        if isinstance(stmt, IfStmt):
            return self._lower_if(stmt, current)
        if isinstance(stmt, ForRangeStmt):
            return self._lower_for(stmt, current)
        if isinstance(stmt, WhileStmt):
            return self._lower_while(stmt, current)
        if isinstance(stmt, ReturnStmt):
            return self._lower_return(stmt, current)
        raise HIRToLIRError(f"Unsupported HIR statement in first LIR lowering pass: {type(stmt).__name__}")

    def _lower_assign(self, stmt: Assign, current: str) -> str:
        if isinstance(stmt.value, Call) and self._is_blocking_call(stmt.value):
            token = self.builder.fresh_token()
            next_block = self.builder.make_block("after_call")
            self.builder.add_op(
                current,
                StartOp(
                    call=stmt.value,
                    result=stmt.target,
                    token=token,
                    source_info=self._origin(stmt.source_info, note="blocking assign"),
                ),
            )
            self.builder.terminate(
                current,
                Await(
                    target=next_block,
                    token=token,
                    source_info=self._origin(stmt.source_info, note="await blocking assign"),
                ),
            )
            return next_block

        self._ensure_pure_expr(stmt.value)
        self.builder.add_op(
            current,
            AssignOp(
                target=stmt.target,
                value=stmt.value,
                source_info=self._origin(stmt.source_info),
            ),
        )
        return current

    def _lower_expr_stmt(self, stmt: ExprStmt, current: str) -> str:
        if not isinstance(stmt.value, Call):
            raise HIRToLIRError("Expected expression statement to be a primitive call")

        if self._is_blocking_call(stmt.value):
            token = self.builder.fresh_token()
            next_block = self.builder.make_block("after_call")
            self.builder.add_op(
                current,
                StartOp(
                    call=stmt.value,
                    token=token,
                    source_info=self._origin(stmt.source_info, note="blocking expr"),
                ),
            )
            self.builder.terminate(
                current,
                Await(
                    target=next_block,
                    token=token,
                    source_info=self._origin(stmt.source_info, note="await blocking expr"),
                ),
            )
            return next_block

        self._ensure_pure_expr(stmt.value)
        self.builder.add_op(
            current,
            SideEffectOp(
                call=stmt.value,
                source_info=self._origin(stmt.source_info, note="comb expr"),
            ),
        )
        return current

    def _lower_if(self, stmt: IfStmt, current: str) -> Optional[str]:
        self._ensure_pure_expr(stmt.cond)

        then_label = self.builder.make_block("if_then")
        merge_label = self.builder.make_block("if_end")
        false_label = self.builder.make_block("if_else") if stmt.else_body else merge_label
        self.builder.terminate(
            current,
            Branch(
                cond=stmt.cond,
                true_target=then_label,
                false_target=false_label,
                source_info=self._origin(stmt.source_info, note="if branch"),
            ),
        )

        merge_reachable = false_label == merge_label

        then_exit = self._lower_stmts(stmt.then_body, then_label)
        if then_exit is not None and self.builder.blocks[then_exit].term is None:
            self.builder.terminate(
                then_exit,
                Jump(merge_label, source_info=self._origin(stmt.source_info, note="if merge")),
            )
            merge_reachable = True

        if stmt.else_body:
            else_exit = self._lower_stmts(stmt.else_body, false_label)
            if else_exit is not None and self.builder.blocks[else_exit].term is None:
                self.builder.terminate(
                    else_exit,
                    Jump(merge_label, source_info=self._origin(stmt.source_info, note="if merge")),
                )
                merge_reachable = True

        return merge_label if merge_reachable else None

    def _lower_while(self, stmt: WhileStmt, current: str) -> str:
        self._ensure_pure_expr(stmt.cond)

        header_label = self.builder.make_block("while_header")
        body_label = self.builder.make_block("while_body")
        exit_label = self.builder.make_block("while_end")

        self.builder.terminate(
            current,
            Jump(header_label, source_info=self._origin(stmt.source_info, note="while enter")),
        )
        self.builder.terminate(
            header_label,
            Branch(
                cond=stmt.cond,
                true_target=body_label,
                false_target=exit_label,
                source_info=self._origin(stmt.source_info, note="while branch"),
            ),
        )

        body_exit = self._lower_stmts(stmt.body, body_label)
        if body_exit is not None and self.builder.blocks[body_exit].term is None:
            self.builder.terminate(
                body_exit,
                Jump(
                    header_label,
                    source_info=self._origin(stmt.source_info, note="while backedge"),
                ),
            )

        return exit_label

    def _lower_for(self, stmt: ForRangeStmt, current: str) -> str:
        self._ensure_pure_expr(stmt.start)
        self._ensure_pure_expr(stmt.stop)

        header_label = self.builder.make_block("for_header")
        body_label = self.builder.make_block("for_body")
        exit_label = self.builder.make_block("for_end")
        loop_index = self._fresh_hidden_var("for_idx", stmt.iter_var.typ)

        self.builder.add_op(
            current,
            AssignOp(
                target=loop_index,
                value=stmt.start,
                source_info=self._origin(stmt.source_info, note="for init"),
            ),
        )
        self.builder.terminate(
            current,
            Jump(header_label, source_info=self._origin(stmt.source_info, note="for enter")),
        )

        cond = Compare(op="<", lhs=VarRef(loop_index), rhs=stmt.stop)
        self.builder.terminate(
            header_label,
            Branch(
                cond=cond,
                true_target=body_label,
                false_target=exit_label,
                source_info=self._origin(stmt.source_info, note="for branch"),
            ),
        )

        self.builder.add_op(
            body_label,
            AssignOp(
                target=stmt.iter_var,
                value=VarRef(loop_index),
                source_info=self._origin(stmt.source_info, note="for iter value"),
            ),
        )
        body_exit = self._lower_stmts(stmt.body, body_label)
        if body_exit is not None and self.builder.blocks[body_exit].term is None:
            step = BinOp(
                op="+",
                lhs=VarRef(loop_index),
                rhs=Const(value=1, typ=loop_index.typ),
            )
            self.builder.add_op(
                body_exit,
                AssignOp(
                    target=loop_index,
                    value=step,
                    source_info=self._origin(stmt.source_info, note="for increment"),
                ),
            )
            self.builder.terminate(
                body_exit,
                Jump(
                    header_label,
                    source_info=self._origin(stmt.source_info, note="for backedge"),
                ),
            )

        return exit_label

    def _lower_return(self, stmt: ReturnStmt, current: str) -> Optional[str]:
        if stmt.value is not None:
            self._ensure_pure_expr(stmt.value)
        self.builder.terminate(
            current,
            Return(value=stmt.value, source_info=self._origin(stmt.source_info)),
        )
        return None

    def _ensure_pure_expr(self, expr: Expr) -> None:
        from .hir import BinOp, Compare, Const, Select, UnaryOp, VarRef

        if isinstance(expr, (Const, VarRef)):
            return
        if isinstance(expr, UnaryOp):
            self._ensure_pure_expr(expr.value)
            return
        if isinstance(expr, (BinOp, Compare)):
            self._ensure_pure_expr(expr.lhs)
            self._ensure_pure_expr(expr.rhs)
            return
        if isinstance(expr, Select):
            self._ensure_pure_expr(expr.cond)
            self._ensure_pure_expr(expr.true_value)
            self._ensure_pure_expr(expr.false_value)
            return
        if isinstance(expr, Call):
            if not expr.func.endswith("_comb"):
                raise HIRToLIRError(
                    f"Blocking primitive call '{expr.func}' cannot appear inside a pure expression context"
                )
            for arg in expr.args:
                self._ensure_pure_expr(arg.value)
            return
        raise HIRToLIRError(f"Unsupported HIR expression in LIR lowering: {type(expr).__name__}")

    def _is_blocking_call(self, call: Call) -> bool:
        return not call.func.endswith("_comb")

    def _fresh_hidden_var(self, prefix: str, typ: Type) -> Var:
        name = f"__{prefix}_{self.next_hidden_var_id}"
        self.next_hidden_var_id += 1
        var = Var(name=name, typ=typ)
        self.hidden_vars.append(var)
        return var

    def _origin(self, source_info: Optional[SourceInfo], note: Optional[str] = None) -> Optional[SourceInfo]:
        if source_info is None and note is None:
            return None
        if source_info is None:
            return SourceInfo(hir_note=note)
        merged_note = source_info.hir_note
        if note:
            merged_note = note if merged_note is None else f"{merged_note}; {note}"
        return replace(source_info, hir_note=merged_note)


def lower_func(func_ir: FuncIR) -> FuncLIR:
    return HIRToLIRLowerer().lower_func(func_ir)


__all__ = ["HIRToLIRError", "HIRToLIRLowerer", "lower_func"]
