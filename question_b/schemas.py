"""
Question B - Input Validation Schemas
Implements strict clinical boundary validation with clear, actionable error messages.
Requirement: "Validate every input with clear error messages (for example, age must be 1 to 120)."
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class HeartPredictionInput(BaseModel):
    """
    Input schema representing clinical parameters for cardiac risk prediction.
    All fields include clinical range validation and descriptive error reporting.
    """
    age: int = Field(
        ...,
        description="Patient age in years (1 to 120)"
    )
    sex: int = Field(
        ...,
        description="Biological sex: 0 = Female, 1 = Male"
    )
    cp: int = Field(
        ...,
        description="Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)"
    )
    trestbps: int = Field(
        ...,
        description="Resting blood pressure in mm Hg on admission (50 to 300)"
    )
    chol: int = Field(
        ...,
        description="Serum cholesterol in mg/dl (100 to 600)"
    )
    fbs: int = Field(
        ...,
        description="Fasting blood sugar > 120 mg/dl (0 = False, 1 = True)"
    )
    restecg: int = Field(
        ...,
        description="Resting ECG results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)"
    )
    thalach: int = Field(
        ...,
        description="Maximum heart rate achieved in bpm (50 to 250)"
    )
    exang: int = Field(
        ...,
        description="Exercise-induced angina (0 = No, 1 = Yes)"
    )
    oldpeak: float = Field(
        ...,
        description="ST depression induced by exercise relative to rest (0.0 to 10.0 mm)"
    )
    slope: int = Field(
        ...,
        description="Slope of peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)"
    )
    ca: int = Field(
        ...,
        description="Number of major vessels (0 to 3) colored by fluoroscopy"
    )
    thal: int = Field(
        ...,
        description="Thalassemia status (3 = normal, 6 = fixed defect, 7 = reversible defect)"
    )

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if v < 1 or v > 120:
            raise ValueError(f"Age must be between 1 and 120 years. Received invalid value: {v}.")
        return v

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"Sex must be 0 (Female) or 1 (Male). Received invalid value: {v}.")
        return v

    @field_validator("cp")
    @classmethod
    def validate_cp(cls, v: int) -> int:
        if v not in (1, 2, 3, 4):
            raise ValueError(f"Chest pain type (cp) must be 1 (typical angina), 2 (atypical angina), 3 (non-anginal), or 4 (asymptomatic). Received: {v}.")
        return v

    @field_validator("trestbps")
    @classmethod
    def validate_trestbps(cls, v: int) -> int:
        if v < 50 or v > 300:
            raise ValueError(f"Resting blood pressure (trestbps) must be within plausible clinical range 50 to 300 mm Hg. Received: {v}.")
        return v

    @field_validator("chol")
    @classmethod
    def validate_chol(cls, v: int) -> int:
        if v < 100 or v > 600:
            raise ValueError(f"Serum cholesterol (chol) must be within clinical range 100 to 600 mg/dl. Received: {v}.")
        return v

    @field_validator("fbs")
    @classmethod
    def validate_fbs(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"Fasting blood sugar (fbs) must be 0 (<= 120 mg/dl) or 1 (> 120 mg/dl). Received: {v}.")
        return v

    @field_validator("restecg")
    @classmethod
    def validate_restecg(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(f"Resting ECG (restecg) must be 0 (normal), 1 (ST-T abnormality), or 2 (hypertrophy). Received: {v}.")
        return v

    @field_validator("thalach")
    @classmethod
    def validate_thalach(cls, v: int) -> int:
        if v < 50 or v > 250:
            raise ValueError(f"Maximum heart rate (thalach) must be between 50 and 250 bpm. Received: {v}.")
        return v

    @field_validator("exang")
    @classmethod
    def validate_exang(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"Exercise induced angina (exang) must be 0 (No) or 1 (Yes). Received: {v}.")
        return v

    @field_validator("oldpeak")
    @classmethod
    def validate_oldpeak(cls, v: float) -> float:
        if v < 0.0 or v > 10.0:
            raise ValueError(f"ST depression (oldpeak) must be between 0.0 and 10.0 mm. Received: {v}.")
        return round(float(v), 2)

    @field_validator("slope")
    @classmethod
    def validate_slope(cls, v: int) -> int:
        if v not in (1, 2, 3):
            raise ValueError(f"Slope must be 1 (upsloping), 2 (flat), or 3 (downsloping). Received: {v}.")
        return v

    @field_validator("ca")
    @classmethod
    def validate_ca(cls, v: int) -> int:
        if v not in (0, 1, 2, 3):
            raise ValueError(f"Major vessels colored by fluoroscopy (ca) must be between 0 and 3. Received: {v}.")
        return v

    @field_validator("thal")
    @classmethod
    def validate_thal(cls, v: int) -> int:
        if v not in (3, 6, 7):
            raise ValueError(f"Thalassemia (thal) must be 3 (normal), 6 (fixed defect), or 7 (reversible defect). Received: {v}.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 55,
                "sex": 1,
                "cp": 4,
                "trestbps": 140,
                "chol": 240,
                "fbs": 0,
                "restecg": 0,
                "thalach": 150,
                "exang": 1,
                "oldpeak": 1.5,
                "slope": 2,
                "ca": 1,
                "thal": 7
            }
        }
    }


class HeartPredictionResponse(BaseModel):
    """Structured response object returned by /predict endpoint."""
    request_id: int
    risk_score: float
    risk_score_percent: str
    prediction: int
    risk_category: str
    plain_words_summary: str
    lifestyle_recommendations: List[str]
    clinical_guidance: str
    timestamp: str


class StatsResponse(BaseModel):
    """Aggregate statistics returned by /stats endpoint."""
    total_requests: int
    average_predicted_risk: float
    average_predicted_risk_percent: str
    high_risk_count: int
    low_risk_count: int
    share_of_high_risk_results: float
    share_of_high_risk_percent: str
    query_engine: str = "Pure Hand-Written SQL (No ORM)"
