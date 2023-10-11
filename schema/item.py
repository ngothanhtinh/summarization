from typing import Union
from pydantic import BaseModel

class Item(BaseModel):
    raw: str
    summarization: Union[str, None] = None