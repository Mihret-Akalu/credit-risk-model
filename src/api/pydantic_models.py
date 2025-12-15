from pydantic import BaseModel

class CustomerData(BaseModel):
    CurrencyCode: str
    ProviderId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    PricingStrategy: int
