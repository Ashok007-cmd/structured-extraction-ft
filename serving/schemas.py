from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Unstructured text to extract structured data from")


class Entity(BaseModel):
    type: str = Field(..., description="Entity type, e.g. 'organization', 'person', 'location'")
    name: str = Field(..., description="Canonical entity name as it appears in the text")


class DateMention(BaseModel):
    raw: str = Field(..., description="Date as it appears verbatim in the text")
    normalized: str = Field(..., description="ISO 8601 normalized date string, e.g. '2026-03-03'")
    context: Optional[str] = Field(None, description="Semantic role of the date, e.g. 'announcement'")


class FinancialFigure(BaseModel):
    type: str = Field(..., description="Financial figure type, e.g. 'funding_raised', 'revenue'")
    amount: float = Field(..., description="Numeric amount in base currency units")
    currency: Optional[str] = Field(None, description="Currency symbol or code, e.g. '$', 'USD'")


class Relationship(BaseModel):
    type: str = Field(..., description="Relationship type, e.g. 'acquired_by', 'invested_in'")
    subject: str = Field(..., description="Subject entity name")
    object: str = Field(..., description="Object entity name")


class Metric(BaseModel):
    name: str = Field(..., description="Metric name, e.g. 'headcount', 'market_share'")
    value: Union[str, float] = Field(..., description="Metric value (numeric or string)")


class ExtractionResult(BaseModel):
    event_type: str = Field(..., description="High-level event category, e.g. 'funding_round', 'acquisition'")
    entities: List[Entity] = Field(default_factory=list, description="Named entities extracted from the text")
    dates: List[DateMention] = Field(default_factory=list, description="Date mentions with ISO normalization")
    financials: List[FinancialFigure] = Field(default_factory=list, description="Financial figures mentioned")
    relationships: List[Relationship] = Field(default_factory=list, description="Entity-to-entity relationships")
    metrics: List[Metric] = Field(default_factory=list, description="Quantitative metrics mentioned")


class ExtractResponse(BaseModel):
    result: Optional[ExtractionResult] = Field(None, description="Parsed structured extraction result; null if output was not valid JSON")
    raw_output: Optional[str] = Field(None, description="Raw model generation string (omitted when EXTRACT_INCLUDE_RAW_OUTPUT=false)")
    valid_json: bool = Field(..., description="Whether the model output was parseable as JSON")
    schema_valid: bool = Field(..., description="Whether the parsed JSON passed schema validation")
    latency_ms: float = Field(..., description="End-to-end inference latency in milliseconds")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Always 'ok' when the process is running")


class ReadyResponse(BaseModel):
    status: str = Field(..., description="'ready' when model is loaded, 'loading' otherwise")
    model_loaded: bool
    model_name_or_path: str
    adapter_path: Optional[str] = None
