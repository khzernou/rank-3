"""Reading the input files and writing the output file, safely."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, List

from pydantic import ValidationError

from src.models import FunctionCall, FunctionCatalog, PromptItem


def _read_json(path: Path, what: str) -> Any:
    """Load one JSON file and turn every failure into a ValueError.

    Args:
        path: The file to read.
        what: A short label used in error messages.

    Returns:
        The decoded JSON value.

    Raises:
        ValueError: If the file is missing, unreadable or not valid JSON.
    """
    if not path.exists():
        raise ValueError(f"{what} file not found: {path}")
    if not path.is_file():
        raise ValueError(f"{what} path is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as err:
        raise ValueError(
            f"{what} file is not valid JSON: {path} "
            f"(line {err.lineno}, column {err.colno}: {err.msg})"
        ) from err
    except UnicodeDecodeError as err:
        raise ValueError(f"{what} file is not UTF-8 text: {path}") from err
    except OSError as err:
        raise ValueError(f"cannot read {what} file {path}: {err}") from err


def _explain(err: ValidationError) -> str:
    """Turn a pydantic error into short readable lines.

    Args:
        err: The validation error raised by pydantic.

    Returns:
        One line per problem, like ``- [0].parameters.a.type: ...``.
    """
    lines = []
    for problem in err.errors():
        where = "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in problem["loc"]
        )
        lines.append(f"  - {where or '(root)'}: {problem['msg']}")
    return "\n".join(lines)


def load_functions(path: Path) -> FunctionCatalog:
    """Read and validate ``functions_definition.json``.

    Args:
        path: Location of the function definitions file.

    Returns:
        The validated catalog of functions.

    Raises:
        ValueError: If the file is missing, broken or has a wrong shape.
    """
    data = _read_json(path, "function definitions")
    if not isinstance(data, list):
        raise ValueError(f"function definitions must be a JSON array: {path}")
    if not data:
        raise ValueError(f"function definitions file is empty: {path}")
    try:
        return FunctionCatalog(functions=data)
    except ValidationError as err:
        raise ValueError(
            f"invalid function definitions in {path}:\n{_explain(err)}"
        ) from err


def load_prompts(path: Path) -> List[PromptItem]:
    """Read and validate ``function_calling_tests.json``.

    Args:
        path: Location of the prompts file.

    Returns:
        The list of prompts, in file order.

    Raises:
        ValueError: If the file is missing, broken or has a wrong shape.
    """
    data = _read_json(path, "prompts")
    if not isinstance(data, list):
        raise ValueError(f"prompts file must be a JSON array: {path}")
    items: List[PromptItem] = []
    for index, entry in enumerate(data):
        try:
            items.append(PromptItem.model_validate(entry))
        except ValidationError as err:
            raise ValueError(
                f"invalid prompt at index {index} in {path}:\n{_explain(err)}"
            ) from err
    return items


def check_output_path(path: Path) -> None:
    """Fail early if the output file could never be written.

    Creates the parent directory (for example ``data/output/``).

    Args:
        path: Where the results will be written.

    Raises:
        ValueError: If the path is a directory or cannot be created.
    """
    if path.is_dir():
        raise ValueError(f"output path is a directory: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ValueError(
            f"cannot create output directory {path.parent}: {err}"
        ) from err


def write_results(path: Path, results: List[FunctionCall]) -> None:
    """Write the results as one JSON array.

    The file is first written to a temporary file and then renamed, so a
    crash halfway never leaves a half-written, invalid JSON file behind.

    Args:
        path: Where to write the results.
        results: One validated function call per prompt.

    Raises:
        ValueError: If the file cannot be written.
    """
    check_output_path(path)
    payload = [item.model_dump() for item in results]
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, indent=2, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
        os.replace(temp_name, path)
    except (OSError, ValueError) as err:
        if temp_name and os.path.exists(temp_name):
            os.remove(temp_name)
        raise ValueError(f"cannot write output file {path}: {err}") from err
