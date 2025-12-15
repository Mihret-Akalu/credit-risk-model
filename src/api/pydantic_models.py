# src/api/pydantic_models.py
from pydantic import BaseModel

class CustomerData(BaseModel):
    Amount: float
    Value: float
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    ProductCategory: str
    ChannelId: str
    ProviderId: str
