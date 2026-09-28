"""Bounded exact numeric/unit and conservative symbolic candidate comparison.

Unknown is never evidence that two answers differ.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from math import isqrt
import time

EQUIVALENT = "established-equivalent"
DISTINCT = "established-distinct"
UNKNOWN = "unknown"

_TOKEN = re.compile(r"\s*(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|[A-Za-z][A-Za-z0-9]{0,7}|[√±{}(),=+\-*/^])")
_MAX_NODES = 64
_MAX_DEPTH = 16
_MAX_TERMS = 64
_MAX_BITS = 512
_CHECK_SECONDS = 0.05

# Factors use m, kg, s as base dimensions; litre is a cubic metre fraction.
_UNITS = {
    "mm": (Fraction(1, 1000), (1, 0, 0)), "cm": (Fraction(1, 100), (1, 0, 0)),
    "m": (Fraction(1), (1, 0, 0)), "km": (Fraction(1000), (1, 0, 0)),
    "mg": (Fraction(1, 1000000), (0, 1, 0)), "g": (Fraction(1, 1000), (0, 1, 0)),
    "kg": (Fraction(1), (0, 1, 0)), "ms": (Fraction(1, 1000), (0, 0, 1)),
    "s": (Fraction(1), (0, 0, 1)), "min": (Fraction(60), (0, 0, 1)),
    "h": (Fraction(3600), (0, 0, 1)), "mL": (Fraction(1, 1000000), (3, 0, 0)),
    "L": (Fraction(1, 1000), (3, 0, 0)),
}
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_UNIT_PART = re.compile(r"(mm|cm|km|mg|kg|min|mL|ms|m|g|s|h|L)(?:\^?([23])|([²³]))?")
_OPAQUE_UNIT = re.compile(
    rf"({_NUMBER})\s*(?:ft|inch|in|yd|mi|lb|oz|nm|µm|μm)(?:\^?[23]|[²³])?(?:\s*/\s*(?:s|min|h))?"
)
_OPAQUE_TEX = re.compile(r"\\(?:sqrt\{\d{1,3}\}|frac\{[+-]?\d{1,3}\}\{[1-9]\d{0,2}\})")


def _number(text):
    if len(text) > 96:
        return None
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    if "/" in text:
        fields = text.split("/")
        if len(fields) != 2 or not all(re.fullmatch(_NUMBER, field.strip()) for field in fields):
            return None
        value = _decimal(fields[0].strip())
        denominator = _decimal(fields[1].strip())
        if value is None or denominator in (None, 0):
            return None
        result = value / denominator
    else:
        if not re.fullmatch(_NUMBER, text):
            return None
        result = _decimal(text)
    return result / 100 if percent and result is not None else result


def _decimal(text):
    exponent = re.search(r"[eE]([+-]?\d+)$", text)
    if exponent and (len(exponent[1]) > 3 or abs(int(exponent[1])) > 12):
        return None
    try:
        value = Decimal(text)
        if not value.is_finite():
            return None
        result = Fraction(value)
        return result if result.numerator.bit_length() <= 512 and result.denominator.bit_length() <= 512 else None
    except (InvalidOperation, ValueError):
        return None


def normalize(text):
    """Return (exact base quantity, dimension) or None on unsupported syntax/resources."""
    if not isinstance(text, str) or len(text) > 160:
        return None
    text = text.strip().replace("−", "-")
    number = re.match(rf"^({_NUMBER}(?:\s*/\s*{_NUMBER})?\s*%?)", text)
    if not number:
        return None
    value = _number(number[1].replace(" ", ""))
    if value is None:
        return None
    units = text[number.end():].strip()
    if not units:
        return value, (0, 0, 0)
    units = re.sub(r"\s+", "", units).replace("·", "*")
    if len(units) > 48:
        return None
    dimensions = [0, 0, 0]
    pos = 0
    denominator = False
    parts = 0
    while pos < len(units):
        if units[pos] in "*/":
            if pos == 0 or pos == len(units) - 1 or units[pos - 1] in "*/":
                return None
            denominator = units[pos] == "/"
            pos += 1
        match = _UNIT_PART.match(units, pos)
        if not match or parts >= 4:
            return None
        factor, dim = _UNITS[match[1]]
        power = int(match[2] or {"²": 2, "³": 3}.get(match[3], 1))
        power *= -1 if denominator else 1
        value *= factor ** power
        for index in range(3):
            dimensions[index] += dim[index] * power
        pos = match.end()
        parts += 1
        if pos < len(units) and units[pos] not in "*/":
            return None
    return (value, tuple(dimensions)) if value.numerator.bit_length() <= 512 and value.denominator.bit_length() <= 512 else None


def compare(left, right, *, context=None):
    """Prove equivalence or universal difference, never infer from samples."""
    first, second = normalize(left), normalize(right)
    if first is not None and second is not None:
        return EQUIVALENT if first == second else DISTINCT
    try:
        deadline = time.monotonic() + _CHECK_SECONDS
        lhs_label, lhs, lhs_kind = _parse(left, deadline)
        rhs_label, rhs, rhs_kind = _parse(right, deadline)
        if lhs_kind != rhs_kind or (lhs_label and rhs_label and lhs_label != rhs_label):
            return UNKNOWN
        if first is not None or second is not None:
            if (first or second)[1] != (0, 0, 0):
                return UNKNOWN
        a = [_polynomial(item, context, deadline) for item in lhs]
        b = [_polynomial(item, context, deadline) for item in rhs]
        if any(item is None for item in (*a, *b)):
            return UNKNOWN
        if len(a) == len(b) and all(any(x == y for y in b) for x in a) and all(
                any(y == x for x in a) for y in b):
            return EQUIVALENT
        if all(_constant(x) is not None for x in (*a, *b)):
            return DISTINCT if {_constant(x) for x in a} != {_constant(y) for y in b} else EQUIVALENT
        if len(a) == len(b) == 1 and _constant(_subtract(a[0], b[0], deadline)) not in (None, 0):
            return DISTINCT
    except (ValueError, ArithmeticError, TimeoutError, RecursionError):
        pass
    return UNKNOWN


def answer_kind(text):
    """Definite scalar versus explicit finite-set answer; None for unsafe syntax."""
    if normalize(text) is not None:
        return "scalar"
    try:
        return _parse(text, time.monotonic() + _CHECK_SECONDS)[2]
    except (ValueError, ArithmeticError, TimeoutError, RecursionError):
        return "scalar" if admit(text) == "unknown" else None


def admit(text, *, context=None):
    """Classify bounded allowed syntax: supported, in-scope unknown, or reject."""
    if normalize(text) is not None:
        return "supported"
    if isinstance(text, str) and len(text) <= 160:
        compact = text.strip()
        unit = _OPAQUE_UNIT.fullmatch(compact)
        if (unit is not None and _number(unit[1]) is not None) or _OPAQUE_TEX.fullmatch(compact):
            return "unknown"  # Vetted fixed-ratio units/LaTeX forms outside the exact checker.
    try:
        deadline = time.monotonic() + _CHECK_SECONDS
        _, items, _ = _parse(text, deadline)
    except (ValueError, ArithmeticError, TimeoutError, RecursionError):
        return "reject"
    try:
        values = [_polynomial(item, context, deadline) for item in items]
        return "supported" if all(value is not None for value in values) else "unknown"
    except (TimeoutError, ArithmeticError):
        return "unknown"  # Syntactically admitted, but the bounded proof was undecided.
    except (ValueError, RecursionError):
        return "reject"


def _tick(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("bounded symbolic comparison exceeded deadline")


def _parse(text, deadline):
    if not isinstance(text, str) or not text.strip() or len(text) > 160:
        raise ValueError("answer length")
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/").replace("²", "^2").replace("³", "^3")
    tokens = []
    offset = 0
    while offset < len(text):
        _tick(deadline)
        match = _TOKEN.match(text, offset)
        if match is None:
            if text[offset:].isspace():
                break
            raise ValueError("disallowed character")
        token = match[0].strip()
        if tokens and match[0] == token and re.fullmatch(r"\d+(?:\.\d*)?", tokens[-1]) and re.fullmatch(r"[A-Za-z]", token):
            tokens.append("*")  # Only adjacent numeric coefficient: 2x, not a unit or function call.
        tokens.append(token)
        if len(tokens) > _MAX_NODES * 2:
            raise ValueError("too many tokens")
        offset = match.end()
    parser = _Parser(tokens, deadline)
    label = None
    if len(tokens) > 2 and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,7}", tokens[0]) and tokens[1] == "=":
        label = tokens[0]
        parser.pos = 2
    set_answer = parser.take("{")
    if set_answer:
        items = [parser.expression()]
        while parser.take(","):
            if len(items) >= 8:
                raise ValueError("set too large")
            items.append(parser.expression())
        parser.expect("}")
    elif parser.take("±"):
        set_answer = True
        value = parser.expression(precedence=2)
        items = [value, parser.node("neg", value)]
    else:
        value = parser.expression()
        items = [value]
        if parser.take("±"):
            set_answer = True
            value = parser.expression(precedence=2)
            items = [parser.node("+", items[0], value), parser.node("-", items[0], value)]
    if parser.pos != len(tokens):
        raise ValueError("trailing tokens")
    return label, items, "set" if set_answer else "scalar"


class _Parser:
    def __init__(self, tokens, deadline):
        self.tokens, self.pos, self.nodes, self.deadline = tokens, 0, 0, deadline

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ""

    def take(self, token):
        if self.peek() == token:
            self.pos += 1
            return True
        return False

    def expect(self, token):
        if not self.take(token):
            raise ValueError(f"expected {token}")

    def node(self, *parts):
        _tick(self.deadline)
        self.nodes += 1
        if self.nodes > _MAX_NODES:
            raise ValueError("too many nodes")
        return parts

    def expression(self, precedence=0, depth=0):
        if depth > _MAX_DEPTH:
            raise ValueError("too deeply nested")
        token = self.peek()
        if not token:
            raise ValueError("missing operand")
        self.pos += 1
        if token in ("+", "-"):
            operand = self.expression(3, depth + 1)
            lhs = operand if token == "+" else self.node("neg", operand)
        elif token in ("√", "sqrt"):
            if token == "sqrt":
                self.expect("(")
                operand = self.expression(depth=depth + 1)
                self.expect(")")
            else:
                operand = self.expression(4, depth + 1)
            lhs = self.node("sqrt", operand)
        elif token == "(":
            lhs = self.expression(depth=depth + 1)
            self.expect(")")
        elif re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,7}", token):
            lhs = self.node("name", token)
        else:
            number = _decimal(token)
            if number is None:
                raise ValueError("invalid number")
            lhs = self.node("number", number)
        while True:
            op = self.peek()
            if op == "^" and precedence <= 4:
                self.pos += 1
                grouped = self.take("(")
                negative = self.take("-")
                if not negative:
                    self.take("+")
                integer = self.peek()
                if not re.fullmatch(r"\d{1,2}", integer) or int(integer) > 8:
                    raise ValueError("bounded integer powers only")
                self.pos += 1
                if grouped:
                    self.expect(")")
                if self.peek() == "^":
                    raise ValueError("ambiguous chained exponent")
                lhs = self.node("power", lhs, -int(integer) if negative else int(integer))
            elif op in ("+", "-", "*", "/") and (1 if op in ("+", "-") else 2) >= precedence:
                self.pos += 1
                priority = 1 if op in ("+", "-") else 2
                lhs = self.node(op, lhs, self.expression(priority + 1, depth + 1))
            else:
                break
        return lhs


def _bound(poly):
    if len(poly) > _MAX_TERMS or any(
            value.numerator.bit_length() > _MAX_BITS or value.denominator.bit_length() > _MAX_BITS
            for value in poly.values()):
        raise ArithmeticError("symbolic arithmetic limit")
    return {key: value for key, value in poly.items() if value}


def _subtract(left, right, deadline):
    _tick(deadline)
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, 0) - value
    return _bound(result)


def _multiply(left, right, deadline):
    result = {}
    for a, x in left.items():
        for b, y in right.items():
            _tick(deadline)
            powers = dict(a)
            for name, power in b:
                powers[name] = powers.get(name, 0) + power
            key = tuple(sorted(powers.items()))
            result[key] = result.get(key, 0) + x * y
            if len(result) > _MAX_TERMS:
                raise ArithmeticError("too many terms")
    return _bound(result)


def _constant(poly):
    return poly.get((), 0) if not any(key for key in poly) else None


def _sign(context, name):
    if not context:
        return None
    text = " ".join(context) if isinstance(context, (tuple, list)) else context
    if not isinstance(text, str) or len(text) > 16000:
        return None
    escaped = re.escape(name)
    # Only a complete condition at the start of a given/question line.
    # In particular, "Without assuming x>0" is not an affirmative assumption.
    prefix = (r"(?:^|\n)\s*(?:(?i:question|given)\s*:\s*)?"
              r"(?:(?i:for|given|assume|assuming|if|where|with|suppose)\s+)?")
    suffix = r"(?=\s*(?:[,;:]|\n|$))"
    matches = re.findall(rf"{prefix}{escaped}\s*(>=|≥|>|<=|≤|<|!=|≠)\s*0{suffix}", text)
    matches += [match.lower() for match in re.findall(
        rf"{prefix}{escaped}\s+(?i:is)\s+(?i:positive|negative)\b{suffix}", text)]
    if len(set(matches)) != 1:
        return None
    return {">": 1, ">=": 3, "≥": 3, "positive": 1,
            "<": -1, "<=": -3, "≤": -3, "negative": -1, "!=": 2, "≠": 2}.get(matches[0])


def _polynomial(node, context, deadline):
    _tick(deadline)
    op = node[0]
    if op == "number":
        return _bound({(): node[1]})
    if op == "name":
        return {((node[1], 1),): Fraction(1)}
    if op == "sqrt":
        operand = node[1]
        if operand[0] == "power" and operand[2] == 2 and operand[1][0] == "name":
            name = operand[1][1]
            sign = _sign(context, name)
            if sign in (1, -1, 3, -3):
                return {((name, 1),): Fraction(1 if sign > 0 else -1)}
        value = _polynomial(operand, context, deadline)
        constant = _constant(value) if value is not None else None
        if constant is None or constant < 0:
            return None
        numerator, denominator = isqrt(constant.numerator), isqrt(constant.denominator)
        return {(): Fraction(numerator, denominator)} if (
            numerator * numerator == constant.numerator and
            denominator * denominator == constant.denominator) else None
    if op == "neg":
        value = _polynomial(node[1], context, deadline)
        return _bound({key: -coefficient for key, coefficient in value.items()}) if value is not None else None
    left = _polynomial(node[1], context, deadline)
    if left is None:
        return None
    if op == "power" and node[2] == 0:
        constant = _constant(left)
        if constant is None:
            if len(left) != 1:
                return None
            (monomial, _), = left.items()
            if len(monomial) != 1 or _sign(context, monomial[0][0]) not in (-1, 1, 2):
                return None
        elif constant == 0:
            return None  # 0^0 is not a universally defined answer.
        return {(): Fraction(1)}
    if op == "power":
        exponent = node[2]
        if exponent < 0:
            constant = _constant(left)
            return _bound({(): constant ** exponent}) if constant not in (None, 0) else None
        result = {(): Fraction(1)}
        for _ in range(exponent):
            result = _multiply(result, left, deadline)
        return result
    right = _polynomial(node[2], context, deadline)
    if right is None:
        return None
    if op == "+":
        return _subtract(left, {key: -value for key, value in right.items()}, deadline)
    if op == "-":
        return _subtract(left, right, deadline)
    if op == "*":
        return _multiply(left, right, deadline)
    denominator = _constant(right)
    if denominator is not None:
        return _multiply(left, {(): 1 / denominator}, deadline) if denominator else None
    if len(right) == 1:
        (monomial, coefficient), = right.items()
        if (len(monomial) == 1 and monomial[0][1] == 1
                and _sign(context, monomial[0][0]) in (-1, 1, 2)):
            name = monomial[0][0]
            if all(dict(key).get(name, 0) for key in left):
                return _bound({tuple((var, power - (var == name)) for var, power in key
                                     if power - (var == name)): value / coefficient
                               for key, value in left.items()})
    return None
