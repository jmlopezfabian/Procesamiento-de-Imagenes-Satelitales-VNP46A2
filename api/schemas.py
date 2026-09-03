"""Pydantic schemas for API request/response."""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from ntl.core.models import MedicionResultado


class JobRequest(BaseModel):
    """Body for POST /jobs."""

    municipios: list[str] = Field(..., min_length=1, description="List of municipality names")
    fecha_inicio: date = Field(..., description="Start date (inclusive)")
    fecha_fin: date = Field(..., description="End date (inclusive)")
    chunks: int | None = Field(None, description="Optional chunk size for processing dates in batches")


class JobStatus(BaseModel):
    """Response for job status (GET /jobs/{job_id})."""

    job_id: str
    status: Literal["pending", "running", "completed", "failed"] = Field(
        ..., description="Current job status"
    )
    progress: str | None = Field(None, description="Human-readable progress e.g. '3/10 fechas'")
    created_at: datetime = Field(..., description="When the job was created")
    finished_at: datetime | None = Field(None, description="When the job finished (if completed/failed)")
    error: str | None = Field(None, description="Error message if status is failed")
    total_results: int = Field(0, description="Number of measurement records when completed")
    fallos: list[dict] = Field(
        default_factory=list,
        description=(
            "Mediciones que no se pudieron calcular por un fallo, con municipio, "
            "fecha, tipo y mensaje. Un job puede terminar 'completed' y aun así "
            "traer filas de menos: si esta lista no está vacía, faltan mediciones "
            "y no es porque no hubiera imagen."
        ),
    )


class JobResult(BaseModel):
    """Response for GET /jobs/{job_id}/results."""

    job_id: str
    results: list[MedicionResultado] = Field(
        ..., description="List of measurement records (MedicionResultado)"
    )


class MunicipiosResponse(BaseModel):
    """Response for GET /municipios."""

    municipios: list[str] = Field(..., description="Available municipality names")


class MatrizRequest(BaseModel):
    """Body for POST /matriz."""

    municipio: str = Field(..., description="Municipality name")
    fecha: date = Field(..., description="Date for the radiance matrix")


class BboxSchema(BaseModel):
    """Bounding box of the cropped matrix."""

    min_x: int = Field(..., description="Minimum x coordinate")
    max_x: int = Field(..., description="Maximum x coordinate")
    min_y: int = Field(..., description="Minimum y coordinate")
    max_y: int = Field(..., description="Maximum y coordinate")


class MatrizResult(BaseModel):
    """Response for GET /matriz/{job_id}/resultado."""

    job_id: str = Field(..., description="Job identifier")
    municipio: str = Field(..., description="Municipality name")
    fecha: date = Field(..., description="Date of the data")
    bbox: BboxSchema = Field(..., description="Bounding box in the original tile")
    rows: int = Field(..., description="Number of rows in the matrices")
    cols: int = Field(..., description="Number of columns in the matrices")
    radiance_matrix: list[list[float | None]] = Field(
        ..., description="Radiance values; NaN/Inf as null"
    )
    municipality_mask: list[list[int]] = Field(
        ..., description="Binary mask: 1=municipality pixel, 0=otherwise"
    )
    municipality_coverage: list[list[float]] = Field(
        ..., description="Fraction of each pixel inside the municipality, in [0, 1]"
    )


class ChatRequest(BaseModel):
    """Body for POST /chat."""

    message: str = Field(..., description="User message")
    history: list[dict] = Field(default_factory=list, description="Message history")


class ChatResponse(BaseModel):
    """Response for POST /chat."""

    response: str = Field(..., description="Agent response text")
    heatmap_data: dict | None = Field(None, description="Radiance matrix data for heatmap")
    mediciones: list[dict] | None = Field(None, description="Measurement records for table")
