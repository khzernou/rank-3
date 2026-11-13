"""Entry point: ``uv run python -m src [options]``.

Reads the function definitions and the prompts, asks the model for one
function call per prompt using constrained decoding, and writes all the
calls to a JSON file.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from src.decoder import ConstrainedDecoder
from src.io_utils import (
    check_output_path,
    load_functions,
    load_prompts,
    write_results,
)
from src.models import CliOptions, FunctionCall
from src.vocabulary import Vocabulary

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calls.json"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_options(argv: Optional[List[str]] = None) -> CliOptions:
    """Read the command line options.

    Args:
        argv: The arguments to parse (defaults to ``sys.argv[1:]``).

    Returns:
        The validated options.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural language requests into function "
                    "calls with constrained decoding.",
    )
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS,
                        help=f"function definitions file "
                             f"(default: {DEFAULT_FUNCTIONS})")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"prompts file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"results file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Hugging Face model name "
                             f"(default: {DEFAULT_MODEL})")
    parser.add_argument("--verbose", action="store_true",
                        help="print each answer while it is generated")
    return CliOptions(**vars(parser.parse_args(argv)))


def log(message: str) -> None:
    """Print a progress message on stderr (stdout stays clean).

    Args:
        message: The text to print.
    """
    print(message, file=sys.stderr, flush=True)


def load_model(name: str) -> ConstrainedDecoder:
    """Load the LLM through the SDK and prepare the decoder.

    Args:
        name: The Hugging Face model identifier.

    Returns:
        A ready ConstrainedDecoder.

    Raises:
        RuntimeError: If the model or its vocabulary cannot be loaded.
    """
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError as err:
        raise RuntimeError(
            "the llm_sdk package is not installed, run `uv sync` first"
        ) from err
    try:
        model = Small_LLM_Model(model_name=name)
        vocab_path = model.get_path_to_vocab_file()
    except Exception as err:
        raise RuntimeError(
            f"could not load the model {name!r}: {err}"
        ) from err
    vocab = Vocabulary.from_file(vocab_path)
    return ConstrainedDecoder(model=model, vocab=vocab)


def run(options: CliOptions) -> int:
    """Do the whole job and return the exit code.

    Args:
        options: The command line options.

    Returns:
        0 on success, 1 if some prompts could not be processed.
    """
    catalog = load_functions(Path(options.functions_definition))
    prompts = load_prompts(Path(options.input))
    output = Path(options.output)
    check_output_path(output)
    log(f"Loaded {len(catalog.functions)} functions and "
        f"{len(prompts)} prompts.")

    if not prompts:
        write_results(output, [])
        log(f"No prompts to process, wrote an empty list to {output}.")
        return 0

    log(f"Loading model {options.model}...")
    decoder = load_model(options.model)

    results: List[FunctionCall] = []
    failures = 0
    started = time.perf_counter()
    for number, item in enumerate(prompts, start=1):
        step_start = time.perf_counter()
        try:
            call = decoder.call(item.prompt, catalog)
        except Exception as err:
            failures += 1
            log(f"[{number}/{len(prompts)}] FAILED {item.prompt!r}: {err}")
            continue
        results.append(call)
        elapsed = time.perf_counter() - step_start
        line = f"[{number}/{len(prompts)}] {elapsed:5.1f}s  {call.name}"
        if options.verbose:
            line += f" {call.parameters}  <- {item.prompt!r}"
        log(line)

    write_results(output, results)
    total = time.perf_counter() - started
    stats = decoder.stats
    log(f"Done: {len(results)}/{len(prompts)} prompts in {total:.1f}s "
        f"({stats.model_calls} model calls, "
        f"{stats.forced_tokens} forced tokens). Output: {output}")
    return 1 if failures else 0


def main(argv: Optional[List[str]] = None) -> int:
    """Run the program and turn every error into a clear message.

    Args:
        argv: Command line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The process exit code.
    """
    try:
        return run(parse_options(argv))
    except (ValueError, RuntimeError, ValidationError) as err:
        log(f"Error: {err}")
    except KeyboardInterrupt:
        log("Interrupted by the user.")
    except MemoryError:
        log("Error: not enough memory to run the model.")
    except Exception as err:
        log(f"Unexpected error ({type(err).__name__}): {err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
