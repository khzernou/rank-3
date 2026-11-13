"""Constrained decoding: the model writes the JSON, but only valid tokens.

How one prompt is processed:

1. We build a chat prompt listing the available functions and the request.
2. The answer always starts with the fixed text ``{"name": "``. Fixed
   text is appended directly (it is the only valid choice anyway).
3. The function name is generated token by token. At each step only the
   tokens that keep the text a prefix of a real function name are
   allowed; all other logits are set to -inf. So the model *chooses* the
   function, but it can only choose one that exists.
4. For each parameter we append ``"key": `` and let the model write the
   value, again restricted to tokens that keep the value valid for its
   type (digits for a number, ``true``/``false`` for a boolean, legal
   JSON string characters for a string).
5. The value ends when the model picks a token that starts the text we
   expect next (``,`` or ``}`` or a closing quote). The model decides
   where the value ends; we decide what is allowed.

Because every token is checked, the answer is always valid JSON that
matches the schema, whatever the model "wanted" to write.
"""

import json
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from src.grammar import (
    BAD,
    CLOSED,
    number_is_complete,
    number_prefix_is_valid,
    number_value,
    scan_string,
    string_value,
)
from src.models import (
    FunctionCall,
    FunctionCatalog,
    FunctionDefinition,
    ParamValue,
)
from src.vocabulary import Vocabulary

SYSTEM_PROMPT = (
    "You are a function-calling engine. You never answer the request "
    "yourself. You choose the one function from the list that does what "
    "the user asks, and you fill in its arguments as JSON.\n"
    "Rules:\n"
    "- Copy text arguments exactly as written in the request, without "
    "the quotes around them.\n"
    "- Write numbers with digits only.\n"
    "- When the request describes characters in words (digits, vowels, "
    "a symbol...), write the actual regular expression or symbol.\n"
    "- Use JSON escaping inside strings (\\\" for a quote, \\\\ for a "
    "backslash).\n\n"
    "Functions:\n"
)


class GenerationStats(BaseModel):
    """Counters used to report how much work the model did."""

    model_calls: int = 0
    forced_tokens: int = 0


class ConstrainedDecoder(BaseModel):
    """Turns a natural language request into a schema-valid function call.

    Attributes:
        model: The ``Small_LLM_Model`` from ``llm_sdk``.
        vocab: Token lookup tables built from the model's vocabulary file.
        max_number_tokens: Safety limit for the length of a number.
        max_string_tokens: Safety limit for the length of a string.
        stats: Counters, reset by the caller when it wants.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    vocab: Vocabulary
    max_number_tokens: int = Field(default=40, ge=1)
    max_string_tokens: int = Field(default=200, ge=1)
    stats: GenerationStats = Field(default_factory=GenerationStats)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        """Tokenize text with the SDK and return a plain list of IDs.

        Args:
            text: Any text.

        Returns:
            The token IDs.
        """
        if not text:
            return []
        ids: List[int] = self.model.encode(text)[0].tolist()
        return ids

    def _force(self, ids: List[int], text: str) -> None:
        """Append text that has only one possible spelling.

        Args:
            ids: The running list of token IDs (modified in place).
            text: The fixed text to add.
        """
        new_ids = self._encode(text)
        self.stats.forced_tokens += len(new_ids)
        ids.extend(new_ids)

    def _pick(self, ids: List[int], allowed: Sequence[int] | NDArray[Any]
              ) -> int:
        """Ask the model for the next token, among ``allowed`` only.

        This is the constrained decoding step: every logit that is not in
        ``allowed`` is replaced by -inf, then we take the highest one
        (greedy decoding). If only one token is allowed, we do not even
        need to ask the model.

        Args:
            ids: The prompt plus everything generated so far.
            allowed: The token IDs that keep the output valid.

        Returns:
            The chosen token ID.

        Raises:
            RuntimeError: If no token is allowed (should never happen).
        """
        choices = np.unique(np.asarray(allowed, dtype=np.int64))
        if choices.size == 0:
            raise RuntimeError("no valid token at this step")
        if choices.size == 1:
            return int(choices[0])
        self.stats.model_calls += 1
        logits = np.asarray(
            self.model.get_logits_from_input_ids(ids), dtype=np.float32
        )
        choices = choices[choices < logits.size]
        if choices.size == 0:
            raise RuntimeError("allowed tokens are outside the model output")
        masked = np.full(logits.shape, -np.inf, dtype=np.float32)
        masked[choices] = logits[choices]
        return int(np.argmax(masked))

    def _close(self, ids: List[int], token: int, token_rest: str,
               follow: str) -> None:
        """Finish a value with a token that starts the expected next text.

        Args:
            ids: Running token IDs (modified in place).
            token: The chosen closing token.
            token_rest: The part of the token text that belongs to
                ``follow`` (for ``",`` this is ``,``).
            follow: The fixed text expected after the value.
        """
        ids.append(token)
        self._force(ids, follow[len(token_rest):])

    # ------------------------------------------------------------------
    # One generator per kind of value
    # ------------------------------------------------------------------

    def _choose_literal(self, ids: List[int], options: List[str],
                        follow: str) -> str:
        """Let the model write one of a fixed set of words.

        Used for the function name and for booleans (``true``/``false``).

        Args:
            ids: Running token IDs (modified in place).
            options: Every allowed word.
            follow: Fixed text expected right after the word.

        Returns:
            The chosen word.
        """
        written = ""
        closing = set(self.vocab.ids_for_prefixes(follow))
        while True:
            matching = [opt for opt in options if opt.startswith(written)]
            if len(matching) == 1:
                # Only one option left: the rest is not a choice anymore.
                self._force(ids, matching[0][len(written):] + follow)
                return matching[0]
            allowed: Set[int] = set()
            for option in matching:
                rest = option[len(written):]
                if rest:
                    allowed.update(self.vocab.ids_for_prefixes(rest))
                else:
                    allowed.update(closing)
            token = self._pick(ids, sorted(allowed))
            text = self.vocab.texts[token] or ""
            if written in options and token in closing:
                self._close(ids, token, text, follow)
                return written
            written += text
            ids.append(token)

    def _write_number(self, ids: List[int], follow: str,
                      integer: bool) -> str:
        """Let the model write a number, digit by digit.

        Args:
            ids: Running token IDs (modified in place).
            follow: Fixed text expected right after the number.
            integer: True for the ``integer`` type.

        Returns:
            The number exactly as written, e.g. ``"265"``.
        """
        written = ""
        closing = self.vocab.ids_for_prefixes(follow)
        for _ in range(self.max_number_tokens):
            allowed = [
                token for token in self.vocab.number_ids
                if number_prefix_is_valid(
                    written + (self.vocab.texts[token] or ""), integer)
            ]
            done = number_is_complete(written, integer)
            if done:
                allowed.extend(closing)
            token = self._pick(ids, allowed)
            text = self.vocab.texts[token] or ""
            if done and token in closing:
                self._close(ids, token, text, follow)
                return written
            written += text
            ids.append(token)
        # Safety limit reached: finish the number ourselves.
        if not number_is_complete(written, integer):
            written = (written + "0"
                       if number_is_complete(written + "0", integer)
                       else "0")
        self._force(ids, follow)
        return written

    def _string_candidates(self, pending: int, follow: str
                           ) -> Tuple[NDArray[np.int64], Dict[int, str]]:
        """List the tokens allowed at the current point of a string.

        Args:
            pending: Escape state (see :func:`src.grammar.scan_string`).
            follow: Fixed text expected after the closing quote.

        Returns:
            ``(continue_ids, closing)`` where ``closing`` maps each token
            that ends the string to the part of it after the quote.
        """
        texts = self.vocab.texts
        if pending == 0:
            # Fast path: "plain" tokens are always fine, only check the
            # few thousand tokens that contain a quote or a backslash.
            to_check: Sequence[int] = self.vocab.special_string_ids
            keep: List[int] = []
            base: NDArray[np.int64] = self.vocab.plain_string_ids
        else:
            # Right after a backslash: every token must be checked.
            to_check = range(len(texts))
            keep = []
            base = np.zeros(0, dtype=np.int64)
        closing: Dict[int, str] = {}
        for token in to_check:
            text = texts[token]
            if not text:
                continue
            status, where = scan_string(text, pending)
            if status == BAD:
                continue
            if status == CLOSED:
                after = text[where + 1:]
                if follow.startswith(after):
                    closing[token] = after
                continue
            keep.append(token)
        extra = np.array(keep + list(closing), dtype=np.int64)
        return np.concatenate([base, extra]), closing

    def _write_string(self, ids: List[int], follow: str) -> str:
        """Let the model write the inside of a JSON string.

        The string ends when the model picks a token containing the
        closing quote (followed by the start of ``follow``).

        Args:
            ids: Running token IDs (modified in place). The opening quote
                must already be there.
            follow: Fixed text expected after the closing quote.

        Returns:
            The decoded string value.
        """
        content = ""
        pending = 0
        for _ in range(self.max_string_tokens):
            allowed, closing = self._string_candidates(pending, follow)
            token = self._pick(ids, allowed)
            text = self.vocab.texts[token] or ""
            if token in closing:
                content += text[:len(text) - len(closing[token]) - 1]
                self._close(ids, token, closing[token], follow)
                return string_value(content)
            _, pending = scan_string(text, pending)
            content += text
            ids.append(token)
        # Safety limit reached: close the string ourselves.
        self._force(ids, '"' + follow)
        return string_value(content)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def build_prompt(catalog: FunctionCatalog, request: str) -> str:
        """Write the chat prompt given to the model.

        Uses the Qwen chat format, with an empty ``<think>`` block so that
        Qwen3 answers directly instead of reasoning first.

        Args:
            catalog: The available functions.
            request: The user's natural language request.

        Returns:
            The full prompt text, ending where the JSON answer starts.
        """
        lines = []
        for function in catalog.functions:
            params = ", ".join(
                f"{name}: {spec.type}"
                for name, spec in function.parameters.items()
            )
            lines.append(f"- {function.name}({params}): "
                         f"{function.description}")
        return (
            "<|im_start|>system\n" + SYSTEM_PROMPT + "\n".join(lines)
            + "<|im_end|>\n"
            + "<|im_start|>user\n" + request + "<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def _write_parameters(self, ids: List[int],
                          function: FunctionDefinition
                          ) -> Dict[str, ParamValue]:
        """Generate every argument of the chosen function, in order.

        Args:
            ids: Running token IDs, ending right after ``"parameters": {``.
            function: The function chosen by the model.

        Returns:
            The arguments with Python types matching the schema.
        """
        values: Dict[str, ParamValue] = {}
        names = list(function.parameters)
        for index, name in enumerate(names):
            spec = function.parameters[name]
            is_last = index == len(names) - 1
            follow = "}}" if is_last else ", "
            if spec.type == "string":
                self._force(ids, f'{_key(name)}: "')
                values[name] = self._write_string(ids, follow)
                continue
            self._force(ids, f"{_key(name)}: ")
            if spec.type == "boolean":
                word = self._choose_literal(ids, ["true", "false"], follow)
                values[name] = word == "true"
            else:
                integer = spec.type == "integer"
                text = self._write_number(ids, follow, integer)
                values[name] = number_value(text, integer)
        if not names:
            self._force(ids, "}}")
        return values

    def call(self, request: str, catalog: FunctionCatalog) -> FunctionCall:
        """Produce the function call for one request.

        Args:
            request: The user's natural language request.
            catalog: The available functions.

        Returns:
            A validated ``FunctionCall`` (prompt, name, parameters).
        """
        ids = self._encode(self.build_prompt(catalog, request))
        self._force(ids, '{"name": "')
        by_name = {function.name: function for function in catalog.functions}
        name = self._choose_literal(
            ids, list(by_name), '", "parameters": {'
        )
        parameters = self._write_parameters(ids, by_name[name])
        return FunctionCall(prompt=request, name=name, parameters=parameters)


def _key(name: str) -> str:
    """Write a parameter name as a JSON key, with quotes and escaping.

    Args:
        name: The parameter name.

    Returns:
        The quoted key, e.g. ``"a"``.
    """
    return json.dumps(name)
