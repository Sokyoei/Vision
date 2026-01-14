from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    name: str = Field(description="tool name")
    args: Optional[Dict[str, Any]] = Field(description="tool input arguments")
