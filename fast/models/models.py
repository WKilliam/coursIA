from pydantic import BaseModel


class Data(BaseModel):
    dayOfWeek: int
    x: float
    y: float
    hour: int