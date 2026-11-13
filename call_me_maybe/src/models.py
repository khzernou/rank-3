"""Pydantic models for everything the program reads, checks and writes."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The JSON types our constrained decoder knows how to generate.
ParamType = Literal["number", "integer", "string", "boolean"]

# A value we can put in the "parameters" object of the output.
ParamValue = Union[float, int, str, bool]


class ParameterSpec(BaseModel):
    """The type of one argument, e.g. ``{"type": "number"}``.

    Extra keys (like a ``description``) are allowed and ignored.
    """

    model_config = ConfigDict(extra="allow")

    type: ParamType


class ReturnSpec(BaseModel):
    """The return type of a function. Kept for information only."""

    model_config = ConfigDict(extra="allow")

    type: str = "unknown"


class FunctionDefinition(BaseModel):
    """One entry of ``functions_definition.json``."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: Optional[ReturnSpec] = None

    @field_validator("name")
    @classmethod
    def name_has_no_quote(cls, value: str) -> str:
        """Reject names that could not appear inside a JSON string as-is.

        Args:
            value: The function name read from the file.

        Returns:
            The unchanged name.

        Raises:
            ValueError: If the name contains a quote, backslash or newline.
        """
        if any(char in value for char in '"\\\n'):
            raise ValueError("function name must not contain quotes, "
                             "backslashes or newlines")
        return value


class FunctionCatalog(BaseModel):
    """The full list of functions the model may choose from."""

    functions: List[FunctionDefinition] = Field(min_length=1)

    @field_validator("functions")
    @classmethod
    def names_are_unique(
        cls, value: List[FunctionDefinition]
    ) -> List[FunctionDefinition]:
        """Make sure two functions never share a name.

        Args:
            value: The parsed function definitions.

        Returns:
            The unchanged list.

        Raises:
            ValueError: If a name appears twice.
        """
        seen: set[str] = set()
        for function in value:
            if function.name in seen:
                raise ValueError(f"duplicate function name: {function.name}")
            seen.add(function.name)
        return value


class PromptItem(BaseModel):
    """One entry of ``function_calling_tests.json``."""

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCall(BaseModel):
    """One entry of the output file: exactly these three keys."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: Dict[str, ParamValue]


class CliOptions(BaseModel):
    """Paths and settings chosen on the command line."""

    functions_definition: str
    input: str
    output: str
    model: str
    verbose: bool = False

    def describe(self) -> Dict[str, Any]:
        """Return the options as a plain dict, for logging.

        Returns:
            A dictionary copy of the options.
        """
        return self.model_dump()
