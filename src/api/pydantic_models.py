from pydantic import BaseModel

class CustomerRFM(BaseModel):
    Recency: float
    Frequency: int
    Monetary: float
