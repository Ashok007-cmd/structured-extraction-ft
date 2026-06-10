from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Unstructured text to extract structured data from")


class Entity(BaseModel):
    type: str
    name: str


class DateMention(BaseModel):
    raw: str
    normalized: str
    context: Optional[str] = None


class FinancialFigure(BaseModel):
    type: str
    amount: float
    currency: Optional[str] = None


class Relationship(BaseModel):
    type: str
    subject: str
    object: str


class Metric(BaseModel):
    name: str
    value: Union[str, float]


class ExtractionResult(BaseModel):
    event_type: str
    entities: List[Entity] = Field(default_factory=list)
    dates: List[DateMention] = Field(default_factory=list)
    financials: List[FinancialFigure] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    metrics: List[Metric] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    result: Optional[ExtractionResult] = None
    raw_output: str
    valid_json: bool
    schema_valid: bool
    latency_ms: float


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name_or_path: str
    adapter_path: Optional[str] = None
