"""FastAPI application exposing the structured-extraction model in real time.

Run locally with:

    uvicorn serving.api:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /v1/extract  - run extraction on a piece of text
    GET  /healthz     - liveness probe (process is up)
    GET  /readyz      - readiness probe (model is loaded)
    GET  /metrics     - Prometheus metrics
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from serving.inference import ExtractionModel
from serving.schemas import (
    ExtractionResult,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    ReadyResponse,
)
from serving.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


_BEARER_DEP = Security(_bearer)


def _check_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = _BEARER_DEP,
) -> None:
    """Enforce Bearer token auth when EXTRACT_API_KEY is configured."""
    settings = request.app.state.settings
    if settings.api_key is None:
        return  # auth disabled
    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


REQUEST_COUNT = Counter(
    "extract_requests_total", "Total number of extraction requests", ["status"]
)
REQUEST_LATENCY = Histogram(
    "extract_request_latency_seconds", "Latency of extraction requests"
)
INFLIGHT_REQUESTS = Gauge(
    "extract_requests_inflight", "Number of extraction requests currently being processed"
)


async def _request_id_middleware(request: Request, call_next):
    """Attach a unique request ID to every request for log correlation."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.model = ExtractionModel(settings)
    # Serialize access to the model: a single loaded model cannot safely run
    # multiple .generate() calls concurrently on one device.
    app.state.inference_semaphore = asyncio.Semaphore(settings.max_concurrency)

    try:
        app.state.model.load()
    except Exception:
        logger.exception("Failed to load model at startup")

    if app.state.model.is_loaded and settings.enable_warmup:
        try:
            logger.info("Running warm-up inference...")
            await asyncio.to_thread(app.state.model.extract, "Warm-up request.")
            logger.info("Warm-up complete")
        except Exception:
            logger.exception("Warm-up inference failed (continuing anyway)")

    yield
    app.state.model.unload()


app = FastAPI(
    title="Structured Extraction API",
    description="Real-time structured JSON extraction from unstructured text using a QLoRA fine-tuned model.",
    version="0.1.0",
    lifespan=lifespan,
)
app.middleware("http")(_request_id_middleware)


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz() -> HealthResponse:
    """Liveness probe — returns 200 as long as the process is running."""
    return HealthResponse(status="ok")


@app.get("/readyz", response_model=ReadyResponse, tags=["ops"])
async def readyz(request: Request) -> ReadyResponse:
    """Readiness probe — returns 200 only once the model is loaded."""
    model: ExtractionModel = request.app.state.model
    settings = request.app.state.settings
    response = ReadyResponse(
        status="ready" if model.is_loaded else "loading",
        model_loaded=model.is_loaded,
        model_name_or_path=settings.model_name_or_path,
        adapter_path=settings.adapter_path,
    )
    if not model.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=response.model_dump())
    return response


@app.get("/metrics", tags=["ops"])
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/extract", response_model=ExtractResponse, tags=["extraction"])
async def extract(
    payload: ExtractRequest,
    request: Request,
    _auth: None = Security(_check_api_key),
) -> ExtractResponse:
    """Extract structured JSON from a piece of unstructured text."""
    model: ExtractionModel = request.app.state.model
    settings = request.app.state.settings

    if not model.is_loaded:
        REQUEST_COUNT.labels(status="unavailable").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet",
        )

    if len(payload.text) > settings.max_request_chars:
        REQUEST_COUNT.labels(status="too_large").inc()
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Input text exceeds max_request_chars={settings.max_request_chars}",
        )

    semaphore: asyncio.Semaphore = request.app.state.inference_semaphore

    with REQUEST_LATENCY.time():
        async with semaphore:
            INFLIGHT_REQUESTS.inc()
            try:
                inference_task = asyncio.to_thread(model.extract, payload.text)
                parsed, raw_output, schema_valid, latency_ms = await asyncio.wait_for(
                    inference_task, timeout=settings.inference_timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error("Extraction timed out after %ss", settings.inference_timeout_seconds)
                REQUEST_COUNT.labels(status="timeout").inc()
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Inference timed out"
                ) from None
            except Exception:
                logger.exception("Extraction failed")
                REQUEST_COUNT.labels(status="error").inc()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Extraction failed"
                ) from None
            finally:
                INFLIGHT_REQUESTS.dec()

    result = None
    if parsed is not None:
        try:
            result = ExtractionResult(**parsed)
        except Exception:
            # Model output was valid JSON but didn't match our response schema;
            # still return the raw JSON for the caller to inspect.
            logger.warning("Extracted JSON did not match ExtractionResult schema")

    REQUEST_COUNT.labels(status="ok" if parsed is not None else "invalid_json").inc()

    return ExtractResponse(
        result=result,
        raw_output=raw_output if settings.include_raw_output else None,
        valid_json=parsed is not None,
        schema_valid=schema_valid,
        latency_ms=latency_ms,
    )
