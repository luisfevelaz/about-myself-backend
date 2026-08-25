from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessageIn] = Field(default_factory=list)
    userReference: Optional[str] = None


class ChatResponse(BaseModel):
    message: str
