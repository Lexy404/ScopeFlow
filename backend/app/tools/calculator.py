"""Arithmetic via a whitelisted AST walk (never `eval`): the model's input is untrusted."""

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from app.llm import ToolSpec
from app.tools.base import ToolError

Number = int | float

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Number]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000  # blocks 9**9**9-style hangs
MAX_ABS_RESULT = 1e100


def _check_size(value: Number) -> Number:
    if abs(value) > MAX_ABS_RESULT:
        raise ToolError("result is too large")
    return value


def _eval(node: ast.AST) -> Number:
    match node:
        case ast.Constant(value=bool()):
            raise ToolError("booleans are not supported")
        case ast.Constant(value=int() | float() as value):
            return value
        case ast.Name(id=name) if name in _CONSTANTS:
            return _CONSTANTS[name]
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY_OPS:
            return _check_size(_UNARY_OPS[type(op)](_eval(operand)))
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BIN_OPS:
            lhs, rhs = _eval(left), _eval(right)
            if isinstance(op, ast.Pow) and abs(rhs) > MAX_EXPONENT:
                raise ToolError(f"exponent larger than {MAX_EXPONENT} is not allowed")
            return _check_size(_BIN_OPS[type(op)](lhs, rhs))
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in _FUNCTIONS:
            return _check_size(_FUNCTIONS[name](*(_eval(a) for a in args)))
        case _:
            raise ToolError(f"unsupported syntax: {ast.dump(node)[:60]}")


def evaluate(expression: str) -> Number:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as e:
        raise ToolError(f"invalid expression: {e.msg}") from e
    try:
        return _eval(tree.body)
    except ZeroDivisionError as e:
        raise ToolError("division by zero") from e
    except (ValueError, TypeError, OverflowError) as e:
        raise ToolError(f"math error: {e}") from e


def _format(value: Number) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}" if isinstance(value, float) else str(value)


class Calculator:
    spec = ToolSpec(
        name="calculator",
        description=(
            "Evaluate an arithmetic expression exactly. Supports + - * / // % ** (or ^), "
            "parentheses, pi, e and sqrt, abs, round, log, log10, exp, sin, cos, tan, "
            "floor, ceil. Use it for any non-trivial calculation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '(17 * 23) / sqrt(2)'"}
            },
            "required": ["expression"],
        },
    )

    async def run(self, args: dict[str, Any]) -> str:
        return _format(evaluate(str(args["expression"])))
