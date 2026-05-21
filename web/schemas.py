from typing import Optional

from pydantic import BaseModel


class ChatImage(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    size: Optional[int] = None
    dataUrl: str

class ChatRequest(BaseModel):
    userId: str
    message: str
    image: Optional[ChatImage] = None

class ChatResponse(BaseModel):
    message: str
