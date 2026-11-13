"""Rules that decide whether a piece of text is still valid JSON.

The decoder writes values one token at a time, so it constantly needs to
know: "if I add this token, can the result still become a valid value?".
These small functions answer that question for numbers and for strings.
"""

import json
import math
import re
from typing import Tuple

# A complete JSON-style number: 12, -3.5, .5, 1e10, 2.5E-3 ...
_FLOAT = re.compile(r"-?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
# A complete integer: 4, -17 ...
_INTEGER = re.compile(r"-?\d+")

# Escape letters allowed after a backslash in a JSON string.
_SIMPLE_ESCAPES = frozenset('"\\/bfnrt')
_HEX = frozenset("0123456789abcdefABCDEF")

# Scan results for strings.
OPEN = "open"      # the string is still going
CLOSED = "closed"  # an unescaped quote ended the string
BAD = "bad"        # the text can never be part of a valid string


def number_is_complete(text: str, integer: bool) -> bool:
    """Tell whether ``text`` is a finished number we could stop on.

    Args:
        text: The digits written so far, e.g. ``"26"`` or ``"1e"``.
        integer: True for the ``integer`` type, False for ``number``.

    Returns:
        True if the text is a whole, finite number.
    """
    pattern = _INTEGER if integer else _FLOAT
    if not pattern.fullmatch(text):
        return False
    return integer or math.isfinite(float(text))


def number_prefix_is_valid(text: str, integer: bool) -> bool:
    """Tell whether ``text`` can still grow into a valid number.

    Trick: every unfinished number ("-", "3.", "1e", "1e+") becomes
    complete if we add one more digit. So we only need to test the text
    itself and the text followed by a ``0``.

    Args:
        text: The candidate text (what we have + one new token).
        integer: True for the ``integer`` type, False for ``number``.

    Returns:
        True if the text is a number or the start of one.
    """
    return (number_is_complete(text, integer)
            or number_is_complete(text + "0", integer))


def number_value(text: str, integer: bool) -> float | int:
    """Convert the generated text into the Python value for the output.

    ``number`` always becomes a float (``2`` is written as ``2.0``), as the
    evaluation requires. ``integer`` becomes an int.

    Args:
        text: A complete number.
        integer: True for the ``integer`` type.

    Returns:
        The converted value.
    """
    if integer:
        return int(text)
    return float(text)


def scan_string(chunk: str, pending: int) -> Tuple[str, int]:
    """Walk through ``chunk`` as if it were inside a JSON string.

    ``pending`` remembers an escape sequence that is not finished yet:
    0 means none, -1 means "just saw a backslash", and 1 to 4 means "still
    waiting for that many hex digits of a \\uXXXX escape".

    Args:
        chunk: The text of one token.
        pending: The escape state before this token.

    Returns:
        ``(OPEN, new_pending)`` if the string continues,
        ``(CLOSED, index)`` if an unescaped quote at ``index`` ends it,
        ``(BAD, 0)`` if the chunk breaks the JSON string rules.
    """
    for index, char in enumerate(chunk):
        if pending == -1:
            if char in _SIMPLE_ESCAPES:
                pending = 0
            elif char == "u":
                pending = 4
            else:
                return BAD, 0
        elif pending > 0:
            if char not in _HEX:
                return BAD, 0
            pending -= 1
        elif char == '"':
            return CLOSED, index
        elif char == "\\":
            pending = -1
        elif ord(char) < 0x20:
            return BAD, 0
    return OPEN, pending


def string_value(content: str) -> str:
    """Turn JSON string content (with escapes) into the real Python text.

    Example: the content ``C:\\\\Users`` becomes ``C:\\Users``.

    Args:
        content: Everything written between the two quotes.

    Returns:
        The decoded text. A broken escape at the very end (only possible
        if generation was cut short) is dropped.
    """
    while True:
        try:
            value = json.loads('"' + content + '"')
            return str(value)
        except json.JSONDecodeError:
            if not content:
                return ""
            content = content[:-1]
