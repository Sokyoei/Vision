from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class Action(BaseModel):
    name: str = Field(description="tool name")
    args: Optional[Dict[str, Any]] = Field(description="tool input arguments")
