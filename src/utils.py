from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class RawOrderItemSchema(BaseModel):
    """Validates raw order item input schema."""

    order_id: str
    product_id: str
    price: float = Field(..., gt=0, description="Price must be greater than 0")
    freight_value: float = Field(..., ge=0, description="Freight value cannot be negative")


class ProcessedDemandSchema(BaseModel):
    """Validates processed aggregated demand feature rows."""

    date: str
    product_category: str
    total_units_sold: int = Field(..., ge=0)
    avg_price: float = Field(..., gt=0)
    total_revenue: float = Field(..., ge=0)

    @field_validator("date")
    def validate_date_format(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return value