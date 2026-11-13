"""The model's dictionary: which text every token ID stands for.

``vocab.json`` maps token text to an ID, but the text is stored in the
GPT-2 "byte-level" alphabet: every raw byte is drawn as a printable
character. A space becomes ``Ġ`` and a newline becomes ``Ċ``. We undo that
mapping so we can ask simple questions such as "is this token a digit?".
"""

import json
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

NUMBER_CHARS = frozenset("0123456789+-.eE")


def byte_level_alphabet() -> Dict[str, int]:
    """Build the table from GPT-2 byte-level characters back to bytes.

    Printable bytes keep their own character. The other bytes (space,
    control characters...) are moved to code points 256 and above, which
    is why a space shows up as ``Ġ`` (code point 288 = 256 + 32).

    Returns:
        A dict mapping each of the 256 characters to the byte it encodes.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table = {chr(byte): byte for byte in printable}
    shift = 0
    for byte in range(256):
        if byte not in printable:
            table[chr(256 + shift)] = byte
            shift += 1
    return table


def token_text(raw: str, alphabet: Dict[str, int]) -> Optional[str]:
    """Translate one vocab entry into the real text it produces.

    Args:
        raw: The key from ``vocab.json`` (e.g. ``"Ġhello"``).
        alphabet: The table built by :func:`byte_level_alphabet`.

    Returns:
        The decoded text (e.g. ``" hello"``), or None when the token is only
        a piece of a multi-byte character and is not valid text by itself.
    """
    try:
        data = bytes(alphabet[char] for char in raw)
        return data.decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


class Vocabulary(BaseModel):
    """Lookup tables built once from ``vocab.json``.

    Attributes:
        texts: ``texts[i]`` is the text of token ``i`` (None if unusable).
        by_text: Reverse lookup, from text to every ID that produces it.
        max_length: Length of the longest token text, in characters.
        plain_string_ids: Tokens that are always safe inside a JSON string
            (no quote, no backslash, no control character).
        special_string_ids: Tokens containing a quote or a backslash. They
            need a closer look before being allowed inside a string.
        number_ids: Tokens made only of characters that appear in numbers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    texts: List[Optional[str]]
    by_text: Dict[str, List[int]]
    max_length: int = Field(ge=1)
    plain_string_ids: NDArray[np.int64]
    special_string_ids: List[int]
    number_ids: List[int]

    @classmethod
    def from_file(cls, path: str) -> "Vocabulary":
        """Load ``vocab.json`` and precompute every lookup table.

        Args:
            path: Path returned by ``get_path_to_vocab_file()``.

        Returns:
            A ready-to-use Vocabulary.

        Raises:
            ValueError: If the file cannot be read or has the wrong shape.
        """
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw_vocab = json.load(handle)
        except (OSError, json.JSONDecodeError) as err:
            raise ValueError(f"cannot read vocabulary {path}: {err}") from err
        if not isinstance(raw_vocab, dict) or not raw_vocab:
            raise ValueError(f"vocabulary file has a wrong shape: {path}")

        alphabet = byte_level_alphabet()
        size = max(int(token_id) for token_id in raw_vocab.values()) + 1
        texts: List[Optional[str]] = [None] * size
        by_text: Dict[str, List[int]] = {}
        plain: List[int] = []
        special: List[int] = []
        numbers: List[int] = []

        for raw, token_id in raw_vocab.items():
            text = token_text(raw, alphabet)
            if not text:
                continue
            token_id = int(token_id)
            texts[token_id] = text
            by_text.setdefault(text, []).append(token_id)
            if all(char in NUMBER_CHARS for char in text):
                numbers.append(token_id)
            if any(ord(char) < 0x20 for char in text):
                continue
            if '"' in text or "\\" in text:
                special.append(token_id)
            else:
                plain.append(token_id)

        return cls(
            texts=texts,
            by_text=by_text,
            max_length=max(len(text) for text in by_text),
            plain_string_ids=np.array(plain, dtype=np.int64),
            special_string_ids=special,
            number_ids=numbers,
        )

    def ids_for_prefixes(self, target: str) -> List[int]:
        """Find every token whose text is a prefix of ``target``.

        Example: for ``"fn_add"`` this returns the IDs of ``"f"``, ``"fn"``,
        ``"fn_"`` and so on, if those tokens exist.

        Args:
            target: The text we want the model to write next.

        Returns:
            The matching token IDs, shortest prefix first.
        """
        found: List[int] = []
        for length in range(1, min(len(target), self.max_length) + 1):
            found.extend(self.by_text.get(target[:length], []))
        return found
