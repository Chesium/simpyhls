from dataclasses import dataclass
from typing import List, Optional


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


@dataclass
class Call(Expr):
    func: str
    args: List[CallArg]


@dataclass
class Stmt:
    pass


@dataclass
class Assign(Stmt):
    target: Var
    value: Expr


@dataclass
class ExprStmt(Stmt):
    value: Expr


@dataclass
class IfStmt(Stmt):
    cond: Expr
    then_body: List[Stmt]
    else_body: List[Stmt]


@dataclass
class ForRangeStmt(Stmt):
    iter_var: Var
    start: Expr
    stop: Expr
    body: List[Stmt]


@dataclass
class WhileStmt(Stmt):
    cond: Expr
    body: List[Stmt]


@dataclass
class ReturnStmt(Stmt):
    value: Optional[Expr]


@dataclass
class FuncIR:
    name: str
    args: List[Var]
    locals: List[Var]
    body: List[Stmt]
