"""Safe arithmetic evaluator (AST based - never uses eval/exec)."""
from __future__ import annotations

import ast
import math
import operator
import statistics

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round, "floor": math.floor,
    "ceil": math.ceil, "min": min, "max": max, "sum": sum, "mean": statistics.mean, "median": statistics.median,
}
_CONSTS = {"pi": math.pi, "e": math.e}
MAX_BITS = 20_000


def _check_size(value):
    if isinstance(value, int) and value.bit_length() > MAX_BITS:
        raise ValueError("Result is too large")
    return value


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numbers are allowed")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"Unknown name '{node.id}'")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Invalid exponent operands")
            if left not in (0, 1, -1) and abs(right) * math.log2(abs(left)) > MAX_BITS:
                raise ValueError("Exponent too large")
        return _check_size(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise ValueError("Unsupported function call")
        return _check_size(_FUNCS[node.func.id](*[_eval(a) for a in node.args]))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def safe_eval(expression: str) -> int | float:
    if not expression or not expression.strip():
        raise ValueError("Empty expression")
    if len(expression) > 500:
        raise ValueError("Expression too long")
    expr = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree)
    except ZeroDivisionError:
        raise ValueError("Division by zero") from None
    except (SyntaxError, TypeError, OverflowError, statistics.StatisticsError) as exc:
        raise ValueError(f"Invalid expression: {exc}") from None
    if isinstance(result, list):
        raise ValueError("Expression must evaluate to a single number")
    return result


def format_number(value: int | float) -> str:
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def calculate(expression: str) -> dict:
    return {"expression": expression, "result": format_number(safe_eval(expression))}
