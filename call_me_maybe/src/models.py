from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

class FunctionParameter(BaseModel):
    """individual param inside function schema definition"""
type: str = Field(..., description="Data type: 'string', 'number', 'integer', 'boolean', 'array', etc.")