from pydantic import BaseModel, Field, validator
from typing import Literal
import config

#para los campos con descripcion y ejemplo
class CustomerData(BaseModel):
    gender: Literal['Male', 'Female'] = Field(
        ...,
        description="Género del cliente",
        example="Female"
    )

    tenure: int = Field(
        ...,
        ge=0,
        le=72,
        description="Meses con la compañía (0-72)",
        example=12
    )

    PhoneService: Literal['Yes', 'No'] = Field(
        ...,
        description="¿Tiene servicio telefónico?",
        example="Yes"
    )

    MultipleLines: Literal['Yes', 'No', 'No phone service'] = Field(
        ...,
        description="¿Tiene múltiples líneas?",
        example="No"
    )

    InternetService: Literal['DSL', 'Fiber optic', 'No'] = Field(
        ...,
        description="Tipo de servicio de internet",
        example="Fiber optic"
    )

    OnlineSecurity: Literal['Yes', 'No', 'No internet service'] = Field(
        ...,
        description="¿Tiene seguridad online?",
        example="No"
    )

    OnlineBackup: Literal['Yes', 'No', 'No internet service'] = Field(
        ...,
        description="¿Tiene respaldo online?",
        example="Yes"
    )

    DeviceProtection: Literal['Yes', 'No', 'No internet service'] = Field(
        ...,
        description="¿Tiene protección de dispositivos?",
        example="No"
    )

    TechSupport: Literal['Yes', 'No', 'No internet service'] = Field(
        ...,
        description="¿Tiene soporte técnico?",
        example="No"
    )

    StreamingTV: Literal['Yes', 'No', 'No internet service'] = Field(
        ...,
        description="¿Tiene streaming de TV?",
        example="Yes"
    )

    StreamingMovies: Literal['Yes', 'No', 'No internet service'] = Field(
        ...,
        description="¿Tiene streaming de películas?",
        example="Yes"
    )

    Contract: Literal['Month-to-month', 'One year', 'Two year'] = Field(
        ...,
        description="Tipo de contrato",
        example="Month-to-month"
    )

    PaymentMethod: Literal[
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ] = Field(
        ...,
        description="Método de pago",
        example="Electronic check"
    )

    TotalCharges: float = Field(
        ...,
        ge=0,
        description="Total de cargos acumulados",
        example=151.65
    )

    #Para un mejor front y mas compresion de los datos
    class Config:
        json_schema_extra = {
            "example": config.EXAMPLE_REQUEST
        }


class ChurnPredictionResponse(BaseModel):
    """Respuesta de predicción de churn"""
    prediction: int = Field(..., description="0 = No churn, 1 = Churn")
    probability: float = Field(..., description="Probabilidad de churn (0.0 - 1.0)")
    churn: str = Field(..., description="'Yes' o 'No'")
    mensaje: str = Field(..., description="Mensaje descriptivo")


class RiskScoreResponse(BaseModel):
    """Respuesta de score de riesgo"""
    probabilidad_churn: float = Field(..., description="Probabilidad calculada (0.0 - 1.0)")
    nivel_riesgo: str = Field(..., description="'Bajo', 'Medio' o 'Alto'")
    mensaje: str = Field(..., description="Mensaje descriptivo")


class CombinedPredictionResponse(BaseModel):
    """Respuesta combinada"""
    churn: ChurnPredictionResponse
    risk_score: RiskScoreResponse