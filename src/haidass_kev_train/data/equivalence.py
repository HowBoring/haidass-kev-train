"""Bounded exact checks for numeric answers with fixed-ratio units.

Unknown is deliberately not distinct: symbolic answers belong to independent adjudication.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re

EQUIVALENT = "established-equivalent"
DISTINCT = "established-distinct"
UNKNOWN = "unknown"

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


def compare(left, right):
    first, second = normalize(left), normalize(right)
    if first is None or second is None:
        return UNKNOWN
    return EQUIVALENT if first == second else DISTINCT
