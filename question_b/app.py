"""
Question B - Web Application Service
FastAPI REST API serving the trained Cardiovascular Risk Prediction Model.
Features:
- POST /predict: Predicts risk probability and returns plain-words summary
- GET /stats: Pure hand-written SQL metrics (Total requests, average risk, high-risk share)
- GET /health: Service liveness and model verification
- Static UI: Interactive Clinical Health Dashboard
"""

import sys
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from config import MODEL_PATH, SCALER_PATH, DB_PATH, FEATURE_NAMES, SEED
from question_b.database import init_db, save_prediction, get_stats, get_recent_predictions
from question_b.schemas import HeartPredictionInput, HeartPredictionResponse, StatsResponse

# Global model holders
model = None
scaler = None


def load_artifacts():
    """Safely loads model and scaler artifacts from disk."""
    global model, scaler
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print(f"[Startup] Loaded trained model from {MODEL_PATH}")
            print(f"[Startup] Loaded scaler from {SCALER_PATH}")
            return True
        except Exception as e:
            print(f"[Startup ERROR] Failed loading model artifacts: {e}")
            model = None
            scaler = None
            return False
    else:
        print(f"[Startup WARNING] Model artifacts not found at {MODEL_PATH}. Prediction endpoint will return 503 until trained.")
        return False


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler for startup and shutdown."""
    init_db()
    load_artifacts()
    yield


# Initialize FastAPI application
app = FastAPI(
    title="Cardiovascular Health Risk Assistant",
    description="Production-grade AI health assessment API for personal wellness and clinical triage.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for open web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for frontend
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom validator to transform raw validation exceptions into clear, actionable error messages.
    Requirement: "Validate every input with clear error messages (for example, age must be 1 to 120)."
    """
    errors = []
    for err in exc.errors():
        field = " -> ".join([str(loc) for loc in err["loc"] if loc != "body"])
        msg = err["msg"]
        # Strip internal Pydantic prefixes
        if "Value error, " in msg:
            msg = msg.replace("Value error, ", "")
        errors.append({
            "field": field,
            "message": msg,
            "input_provided": err.get("input")
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "validation_error",
            "error_count": len(errors),
            "message": "Input validation failed. Please correct the clinical fields indicated below.",
            "details": errors
        }
    )


def determine_risk_category(probability: float) -> Dict[str, Any]:
    """Translates numeric probability into clinical plain-words interpretation and guidance."""
    if probability < 0.30:
        return {
            "category": "Low Risk",
            "plain_words": (
                f"Your cardiovascular risk index is {round(probability * 100, 1)}%, which is considered Low. "
                "Your clinical markers do not currently indicate significant coronary artery disease."
            ),
            "lifestyle": [
                "Maintain regular aerobic exercise (150 minutes of moderate activity per week).",
                "Adhere to a heart-healthy diet rich in whole grains, fiber, and lean proteins.",
                "Continue routine annual health screenings for blood pressure and lipids."
            ],
            "guidance": "Favorable risk profile. Continue standard preventive care with your primary physician."
        }
    elif probability < 0.60:
        return {
            "category": "Moderate Risk",
            "plain_words": (
                f"Your cardiovascular risk index is {round(probability * 100, 1)}%, which falls into the Moderate (Borderline) category. "
                "Certain clinical indicators (such as blood pressure, cholesterol, or resting ECG) show early signs of cardiovascular strain."
            ),
            "lifestyle": [
                "Reduce dietary sodium (< 2,300 mg/day) and limit saturated and trans fats.",
                "Engage in structured cardiovascular exercise with physician clearance.",
                "Monitor resting blood pressure at home weekly."
            ],
            "guidance": "Clinical monitoring advised. Schedule a routine consult with your physician for an ECG review."
        }
    else:
        return {
            "category": "High Risk",
            "plain_words": (
                f"Your cardiovascular risk index is {round(probability * 100, 1)}%, indicating High Risk. "
                "Multiple diagnostic indicators (such as fluoroscopy vessel count, ST depression, or chest pain) "
                "correlate strongly with coronary artery narrowing (>50% stenosis)."
            ),
            "lifestyle": [
                "Avoid strenuous or sudden physical exertion until thoroughly evaluated by a cardiologist.",
                "Maintain strict adherence to any prescribed blood pressure or cholesterol medications.",
                "Seek immediate emergency care if you experience chest tightness, radiation of pain to the left arm or jaw, or shortness of breath."
            ],
            "guidance": "Urgent clinical consultation recommended. Follow up promptly with a cardiologist for diagnostic stress testing and coronary imaging."
        }


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the interactive frontend web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>Cardiovascular Health Risk Assistant API is Running.</h2><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>")


@app.post("/predict", response_model=HeartPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_risk(patient: HeartPredictionInput, request: Request):
    """
    Level 1 & Level 2 /predict endpoint:
    - Ingests validated patient clinical data.
    - Computes risk probability using Question A model.
    - Returns plain-words risk interpretation.
    - Persists request and results in SQLite via hand-written SQL.
    """
    global model, scaler
    if model is None or scaler is None:
        # Attempt reloading if files were generated after app started
        if not load_artifacts():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained model artifact not available on server. Please train the model via Question A first."
            )

    # Convert patient input into feature DataFrame
    input_dict = patient.model_dump()
    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]

    try:
        # Scale features and run inference
        scaled_vector = scaler.transform(input_df)
        probability = float(model.predict_proba(scaled_vector)[0, 1])
        prediction = int(probability >= 0.50)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution error: {str(e)}"
        )

    # Derive clinical plain-words classification
    interpretation = determine_risk_category(probability)
    user_agent = request.headers.get("user-agent", "Unknown Client")

    # Persist in SQLite using hand-written parameterized SQL
    try:
        req_id = save_prediction(
            input_data=input_dict,
            risk_score=probability,
            prediction=prediction,
            risk_category=interpretation["category"],
            user_agent=user_agent
        )
    except Exception as e:
        print(f"[Database Error]: Failed saving prediction: {e}")
        req_id = -1

    return HeartPredictionResponse(
        request_id=req_id,
        risk_score=round(probability, 4),
        risk_score_percent=f"{round(probability * 100, 2)}%",
        prediction=prediction,
        risk_category=interpretation["category"],
        plain_words_summary=interpretation["plain_words"],
        lifestyle_recommendations=interpretation["lifestyle"],
        clinical_guidance=interpretation["guidance"],
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


@app.get("/stats", response_model=StatsResponse)
async def get_system_stats():
    """
    Level 2 /stats endpoint:
    Executes hand-written SQL (no ORM) to return aggregate health metrics:
    - Total requests
    - Average predicted risk
    - Share of high-risk results
    """
    try:
        stats_data = get_stats()
        return StatsResponse(**stats_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query database stats: {str(e)}"
        )


@app.get("/recent")
async def get_recent(limit: int = 5):
    """Returns recent predictions for live dashboard updates."""
    try:
        return get_recent_predictions(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Liveness & health check verifying model status and candidate seed."""
    return {
        "status": "healthy",
        "service": "Cardiovascular Health Risk Assistant",
        "candidate_seed": SEED,
        "model_loaded": model is not None,
        "database": str(DB_PATH),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("question_b.app:app", host="127.0.0.1", port=8000, reload=False)
