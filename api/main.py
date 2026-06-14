"""FastAPI application for real-time credit risk inference."""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference_service import get_inference_service
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionItem,
    CreditApplication,
    FeatureImportanceItem,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = get_inference_service()
    app.state.inference = service
    yield


app = FastAPI(
    title="Credit Risk Inference API",
    description=(
        "Real-time credit default prediction — reduces manual risk analysis "
        "effort by automating scoring, batch review, and portfolio triage."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_response(result) -> PredictionResponse:
    prob = result.default_probability
    return PredictionResponse(
        default_probability=prob,
        default_probability_pct=round(prob * 100, 2),
        risk_category=result.risk_category,
        recommendation=result.recommendation,
        latency_ms=round(result.latency_ms, 2),
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    service = get_inference_service()
    return HealthResponse(
        status="ok" if service.is_ready else "degraded",
        model_ready=service.is_ready,
    )


@app.get("/api/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info() -> ModelInfoResponse:
    service = get_inference_service()
    info = service.get_model_info()
    importance = service.get_feature_importance()
    return ModelInfoResponse(
        **info,
        feature_importance=[
            FeatureImportanceItem(**item) for item in importance
        ],
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(application: CreditApplication) -> PredictionResponse:
    service = get_inference_service()
    try:
        result = service.predict_one(application.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _to_response(result)


@app.post("/api/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    service = get_inference_service()
    records = [app.model_dump() for app in request.applications]

    start = time.perf_counter()
    try:
        raw = service.predict_batch(records)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    total_ms = (time.perf_counter() - start) * 1000

    predictions = [
        BatchPredictionItem(
            index=item["index"],
            default_probability=item["default_probability"],
            default_probability_pct=round(item["default_probability"] * 100, 2),
            risk_category=item["risk_category"],
            recommendation=item["recommendation"],
            latency_ms=round(item["latency_ms"], 2),
        )
        for item in raw
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        portfolio_summary=service.portfolio_summary(raw),
        total_latency_ms=round(total_ms, 2),
    )


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """Stream real-time predictions as form fields change."""
    await websocket.accept()
    service = get_inference_service()

    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("action") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            try:
                application = CreditApplication(**payload)
                result = service.predict_one(application.model_dump())
                response = _to_response(result)
                await websocket.send_json(
                    {"type": "prediction", "data": response.model_dump()}
                )
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass


@app.get("/", include_in_schema=False)
def dashboard():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(index)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
