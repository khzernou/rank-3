*This project has been created as part of the 42 curriculum by khzernou.*

# call me maybe

## Description

`call_me_maybe` turns a natural language request into a structured function
call. Given the request *"What is the sum of 2 and 3?"* and a list of
available functions, it does not answer `5`. It writes the call that
should be made:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

It uses the small **Qwen/Qwen3-0.6B** model through the provided `llm_sdk`.
Small models are unreliable at writing JSON on their own, so the program
uses **constrained decoding**: at every step of generation, every token
that would break the JSON or the schema gets a score of `-inf`, so the
model can only pick valid tokens. The model still makes the real
decisions (which function, which values), but the output is always valid
JSON that matches `functions_definition.json`.

## Instructions

Requirements: Python 3.10 or later and [uv](https://docs.astral.sh/uv/).

```bash
make install   # or: uv sync
make run       # or: uv run python -m src
```

The first run downloads the model (about 1.5 GB) into the Hugging Face
cache. On 42 computers, point the cache to goinfre first:
`export HF_HOME=/goinfre/$USER/hf`.

Other Makefile rules:

| Rule | What it does |
|---|---|
| `make debug` | runs the program inside `pdb` |
| `make lint` | `flake8 .` and `mypy` with the flags required by the subject |
| `make lint-strict` | `flake8 .` and `mypy --strict` |
| `make clean` | removes `__pycache__`, `.mypy_cache` and `.pytest_cache` |

### Example usage

```bash
# Default paths: data/input/*.json -> data/output/function_calls.json
uv run python -m src

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json

# Print every answer while it is generated
uv run python -m src --verbose

# Try another model (the project is built and tested for Qwen3-0.6B)
uv run python -m src --model Qwen/Qwen3-1.7B
```

Progress and errors go to stderr. The output directory is created if
needed. If an input file is missing, is not valid JSON, or has the wrong
shape, the program prints a clear message and exits with code 1 instead
of crashing, for example:

```
Error: prompts file is not valid JSON: data/input/x.json (line 2, column 1: Expecting ',' delimiter)
Error: invalid function definitions in defs.json:
  - .functions[0].parameters.x.type: Input should be 'number', 'integer', 'string' or 'boolean'
```

## Algorithm explanation

### 1. The generation loop

An LLM only predicts one token at a time.
`get_logits_from_input_ids(ids)` returns one score (logit) for each of the
151,936 possible tokens. We pick a token, append its ID, and ask again.

### 2. Knowing what each token says (`src/vocabulary.py`)

To decide whether a token is allowed, we need its text. `vocab.json`
(from `get_path_to_vocab_file()`) maps token text to ID, but in the GPT-2
byte-level alphabet, where a space is written `Ġ` and a newline `Ċ`. We
rebuild that alphabet, convert every entry back to real text, and build
lookup tables once at start-up:

- `texts[id]`: the real text of each token;
- `by_text`: text to IDs, used to find "every token that is a prefix of X";
- `number_ids`: tokens made only of `0-9 + - . e E`;
- `plain_string_ids`: tokens that are always safe inside a JSON string
  (no `"`, no `\`, no control character), stored as a numpy array;
- `special_string_ids`: tokens with a quote or backslash, checked one by
  one when needed.

Tokens that are only half of a multi-byte UTF-8 character are left out.

### 3. The constrained step (`ConstrainedDecoder._pick`)

```
allowed = tokens that keep the output valid
if only one token is allowed: take it (no model call needed)
logits = model.get_logits_from_input_ids(ids)
masked = [-inf] * len(logits); masked[allowed] = logits[allowed]
next token = argmax(masked)     # greedy decoding
```

### 4. Writing the answer piece by piece (`src/decoder.py`)

The answer always has the shape

```
{"name": "<function>", "parameters": {"<key1>": <value1>, "<key2>": <value2>}}
```

The structural parts (`{"name": "`, `", "parameters": {`, `"a": `, ...) have
only one valid spelling, so they are appended directly ("forced"). The
model is asked only where there is a real choice:

- **Function name**: allowed tokens are those whose text keeps what was
  written a prefix of at least one function name. As soon as only one
  name is still possible, the rest is forced. So the LLM chooses the
  function, but can never invent one.
- **number / integer**: allowed tokens keep the text a valid number
  prefix (`grammar.number_prefix_is_valid`). Trick: any unfinished number
  (`-`, `3.`, `1e`) becomes complete by adding a `0`, so a text is a
  valid prefix if it, or it plus `0`, fully matches the number pattern.
  `number` values are written as floats (`2.0`), `integer` values as ints.
- **string**: `grammar.scan_string` follows the JSON string rules,
  including escapes (`\"`, `\\`, `\n`, `\uXXXX`). The string can contain
  anything legal, so the model can copy `C:\Users\john` (written
  `C:\\Users\\john`) or `Say "hello"`.
- **boolean**: the model chooses between the two words `true` and `false`.

**How a value ends**: the model decides. At each step, besides the tokens
that continue the value, we also allow tokens that *start the expected
next text* (for a number followed by another argument, `,` or `, `; for
the last string argument, `"` or `"}`...). When the model picks one of
those, the value is finished and the rest of the fixed text is forced.

Each value also has a safety limit on its length (40 tokens for numbers,
200 for strings), so generation always finishes.

Finally the values are converted to Python types, checked again by the
`FunctionCall` pydantic model, and written with `json.dump`, so the output
file is always valid JSON with exactly `prompt`, `name` and `parameters`.

## Design decisions

- **One pass per prompt, not two.** The name and the arguments are
  generated in the same conversation, so the model sees its own choice of
  function when it writes the arguments, and the prompt is processed once.
- **Force everything that is not a choice.** Fixed JSON text and the end
  of a function name after it becomes unique never cost a model call.
  This is the main speed optimisation, because the SDK has no cache and
  every call re-reads the whole prompt.
- **Greedy decoding.** For function calling we want the single most
  likely valid answer, and greedy makes runs reproducible.
- **Short Qwen chat prompt** listing each function as
  `name(param: type): description`, with an empty `<think></think>` block
  so Qwen3 answers directly instead of reasoning first.
- **Supported types**: `number`, `integer`, `string`, `boolean`. Any other
  type in `functions_definition.json` is rejected with a clear message at
  start-up instead of producing output that might not match the schema.
- **Pydantic for every class**: input files, options, output entries,
  the vocabulary and the decoder itself. Errors use the built-in
  `ValueError`/`RuntimeError`, turned into one readable message in `main`.
- **Safe output writing**: results go to a temporary file which is then
  renamed, so an interrupted run never leaves a broken JSON file.
- **Only public SDK methods** (`encode`, `get_logits_from_input_ids`,
  `get_path_to_vocab_file`). The project never imports torch or
  transformers itself.

## Performance analysis

- **JSON validity: 100% by construction.** Every token is checked, and the
  file is written by `json.dump` with `allow_nan=False`. This was tested
  with a fake model returning *random* scores: every output was still
  valid, with the right function names, keys and types.
- **Grammar coverage.** With a fake model that tries to write the
  expected answers of the public and private test sets, 22/22 answers
  were produced exactly, including escaped quotes, Windows paths and
  regular expressions. This shows the constraints never block a correct
  answer.
- **Accuracy with Qwen3-0.6B**: measured with the moulinette on the real
  model. Fill in your own run here: public `__/11`, private `__/11`.
- **Speed.** The decoder's own work is about 3 ms per step; the time is
  spent in the model. A prompt needs roughly 10 to 30 model calls
  (the digits and characters of the values, plus one call to end each
  value). Measured run time for 11 prompts: `__ s` (fill in).

## Challenges faced

- **Token text is not plain text.** `vocab.json` stores the byte-level
  form (`Ġhello`), so the GPT-2 byte alphabet had to be rebuilt to know
  what a token really writes.
- **Knowing where a value stops.** Forcing a fixed length would be
  wrong, so a value ends when the model chooses a token that begins the
  text expected next.
- **Escapes inside strings.** A backslash changes the meaning of the next
  character, so the string checker keeps a small state between tokens.
- **Speed.** Checking 151,936 tokens one by one in Python at each step is
  slow. Tokens that are always safe in strings are precomputed in a numpy
  array, and only the few thousand tokens with a quote or backslash are
  checked at run time.
- **`2` vs `2.0`.** The evaluation requires floats for `number`, so values
  are converted after generation according to the schema.
- **The grader matches answers by position**, so the program always
  writes the results in the same order as the prompts.

## Testing strategy

- `make lint` and `make lint-strict` (flake8, mypy, mypy --strict).
- Error cases: missing files, invalid JSON, non-UTF-8 files, a JSON object
  instead of an array, an empty function list, an unsupported parameter
  type, a prompt without the `prompt` key, an output path that is a
  directory, unknown command line options, an empty prompt list.
- Edge prompts: empty string, unicode and emoji, quotes and backslashes,
  very large and negative numbers, a function with no parameters, a
  boolean parameter, functions with three parameters.
- A fake SDK using the real Qwen tokenizer with random logits (validity)
  and with "oracle" logits (grammar coverage), so the logic could be tested
  without the model weights.
- The provided moulinette, on the public and private sets, with the real
  model:

```bash
cd moulinette && uv sync
uv run python -m moulinette prepare_exercises --set private
cd ../call_me_maybe
uv run python -m src --functions_definition ../moulinette/data/input/functions_definition.json \
  --input ../moulinette/data/input/function_calling_tests.json \
  --output ../moulinette/data/input/function_calls.json
cd ../moulinette
uv run python -m moulinette grade_student_answers --set private \
  --student_answer_path data/input/function_calls.json
```

## Resources

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) and the Qwen
  chat template.
- [Hugging Face: Byte-Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)
- [OpenAI GPT-2 `encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
  (the `bytes_to_unicode` table behind the `Ġ` character).
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
  for numbers, strings and escapes.
- Willard & Louf, *Efficient Guided Generation for Large Language Models*
  (2023), the idea behind constrained decoding.
- [Pydantic documentation](https://docs.pydantic.dev/) and the
  [uv documentation](https://docs.astral.sh/uv/).

### How AI was used

Claude (Anthropic) was used as a tutor and coding assistant: to explain
tokens, logits and constrained decoding step by step, to help write and
review the code of `src/`, to design the edge-case tests, and to draft
this README. All the code was read, run and checked by me, and I can
explain every part of it.
