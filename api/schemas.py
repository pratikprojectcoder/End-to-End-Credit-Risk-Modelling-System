"""Pydantic schemas for the credit risk inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    loan_amnt: float = Field(..., ge=1000, le=100000, description="Requested loan amount")
    annual_inc: float = Field(..., ge=1000, description="Annual income")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income ratio (%)")
    fico_range_low: float = Field(..., ge=300, le=850, description="FICO score")
    revol_bal: float = Field(..., ge=0, description="Revolving balance")
    installment: float = Field(..., ge=0, description="Monthly installment")
    delinq_2yrs: float = Field(0, ge=0, le=20, description="Delinquencies in past 2 years")
    pub_rec: float = Field(0, ge=0, le=10, description="Derogatory public records")


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., ge=0, le=1)
    default_probability_pct: float
    risk_category: Literal["LOW", "MEDIUM", "HIGH"]
    recommendation: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    applications: list[CreditApplication] = Field(..., min_length=1, max_length=1000)


class BatchPredictionItem(BaseModel):
    index: int
    default_probability: float
    default_probability_pct: float
    risk_category: Literal["LOW", "MEDIUM", "HIGH"]
    recommendation: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    portfolio_summary: dict
    total_latency_ms: float


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class ModelInfoResponse(BaseModel):
    model_type: str
    feature_count: int
    feature_columns: list[str]
    risk_thresholds: dict[str, float]
    predictions_served: int
    metrics: dict[str, float]
    feature_importance: list[FeatureImportanceItem]


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    version: str = "1.0.0"
