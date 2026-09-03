"""FastAPI routes for the Black Marble (VNP46A2) processing API."""
import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, HTTPException

from ntl.core.config import PIXELES_MUNICIPIOS
from ntl.core.models import MedicionResultado
from ntl.core.utils import normalize_municipio

from .agent import get_agent, get_last_tool_results
from .job_manager import job_store, run_job, run_matriz_job
from .schemas import (
    ChatRequest,
    ChatResponse,
    JobRequest,
    JobResult,
    JobStatus,
    MatrizRequest,
    MatrizResult,
    MunicipiosResponse,
)

router = APIRouter()


def _get_available_municipios() -> list[str]:
    """Load municipality names from config JSON path."""
    with open(PIXELES_MUNICIPIOS, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys())


def _build_fechas(fecha_inicio, fecha_fin) -> list[str]:
    """Build list of date strings dd-mm-yy from inicio to fin (inclusive)."""
    fechas = []
    d = fecha_inicio
    while d <= fecha_fin:
        fechas.append(d.strftime("%d-%m-%y"))
        d += timedelta(days=1)
    return fechas


@router.get("/municipios", response_model=MunicipiosResponse)
async def list_municipios():
    """List available municipality names for processing."""
    municipios = _get_available_municipios()
    return MunicipiosResponse(municipios=municipios)


@router.post("/jobs", response_model=JobStatus, status_code=202)
async def create_job(body: JobRequest):
    """Create a new processing job. Returns immediately with job_id; poll GET /jobs/{job_id} for status."""
    available = _get_available_municipios()
    normalized = [m.lower().strip() for m in body.municipios]
    invalid = [m for m in normalized if m not in available]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Municipios not available: {invalid}. Use GET /municipios for the list.",
        )
    if body.fecha_fin < body.fecha_inicio:
        raise HTTPException(status_code=400, detail="fecha_fin must be >= fecha_inicio")

    fechas = _build_fechas(body.fecha_inicio, body.fecha_fin)
    if not fechas:
        raise HTTPException(status_code=400, detail="No dates in range")

    job_id = str(uuid.uuid4())
    state = job_store.create(job_id)
    task = asyncio.create_task(
        run_job(job_id, normalized, fechas, body.chunks)
    )
    job_store.set_task(job_id, task)

    return JobStatus(
        job_id=job_id,
        status=state.status,
        progress=state.progress,
        created_at=state.created_at,
        finished_at=state.finished_at,
        error=state.error,
        total_results=state.total_results,
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get current status and progress of a job."""
    state = job_store.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=state.job_id,
        status=state.status,
        progress=state.progress,
        created_at=state.created_at,
        finished_at=state.finished_at,
        error=state.error,
        total_results=state.total_results,
        fallos=getattr(state, "fallos", []),
    )


@router.get("/jobs/{job_id}/results", response_model=JobResult)
async def get_job_results(job_id: str):
    """Get results of a completed job. Returns 409 if job is not yet completed."""
    state = job_store.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Job not finished (status: {state.status}). Poll GET /jobs/{job_id}.",
        )
    if state.status == "failed":
        raise HTTPException(
            status_code=409,
            detail=f"Job failed: {state.error or 'Unknown error'}",
        )
    results = [MedicionResultado.model_validate(r) for r in state.results]
    return JobResult(job_id=job_id, results=results)


@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_job(job_id: str):
    """Cancel a pending or running job and remove it from the store."""
    state = job_store.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.task and not state.task.done():
        state.task.cancel()
        try:
            await state.task
        except asyncio.CancelledError:
            pass
    job_store.remove(job_id)


# --- Matriz endpoints ---


@router.post("/matriz", response_model=JobStatus, status_code=202)
async def create_matriz_job(body: MatrizRequest):
    """Create a matriz extraction job. Returns immediately with job_id; poll GET /matriz/{job_id} for status."""
    available = _get_available_municipios()
    normalized = normalize_municipio(body.municipio)
    if normalized not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Municipio '{body.municipio}' not available. Use GET /municipios for the list.",
        )

    job_id = str(uuid.uuid4())
    state = job_store.create(job_id)
    task = asyncio.create_task(run_matriz_job(job_id, body.municipio, body.fecha))
    job_store.set_task(job_id, task)

    return JobStatus(
        job_id=job_id,
        status=state.status,
        progress=state.progress,
        created_at=state.created_at,
        finished_at=state.finished_at,
        error=state.error,
        total_results=state.total_results,
    )


@router.get("/matriz/{job_id}", response_model=JobStatus)
async def get_matriz_job_status(job_id: str):
    """Get current status and progress of a matriz job."""
    state = job_store.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=state.job_id,
        status=state.status,
        progress=state.progress,
        created_at=state.created_at,
        finished_at=state.finished_at,
        error=state.error,
        total_results=state.total_results,
    )


def _history_to_messages(history: list[dict]) -> list:
    """Convert simplified frontend history to PydanticAI ModelMessage list."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        RequestUsage,
        TextPart,
        UserPromptPart,
    )

    now = datetime.now(timezone.utc)
    messages = []
    for h in history:
        role = (h.get("role") or "").lower()
        content = h.get("content") or ""
        if not content:
            continue
        if role == "user":
            messages.append(
                ModelRequest(
                    parts=[UserPromptPart(content=content, timestamp=now)],
                    timestamp=now,
                )
            )
        elif role in ("model", "assistant"):
            messages.append(
                ModelResponse(
                    parts=[TextPart(content=content)],
                    usage=RequestUsage(),
                    timestamp=now,
                )
            )
    return messages


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """Run the PydanticAI agent and return response with optional heatmap/mediciones."""
    print(f"[Chat] Received message: {body.message[:80]!r}{'...' if len(body.message) > 80 else ''}")
    print(f"[Chat] History: {len(body.history)} messages")
    agent = get_agent()
    message_history = _history_to_messages(body.history) if body.history else None

    try:
        print("[Chat] Running agent.run()...")
        result = await agent.run(body.message, message_history=message_history or None)
        print("[Chat] Agent completed successfully")
    except Exception as e:
        print(f"[Chat] Agent error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    response_text = str(result.output) if result.output is not None else ""
    heatmap_data, mediciones = get_last_tool_results()
    print(f"[Chat] Response: {len(response_text)} chars, heatmap={heatmap_data is not None}, mediciones={mediciones is not None} ({len(mediciones or [])} rows)")

    return ChatResponse(
        response=response_text,
        heatmap_data=heatmap_data,
        mediciones=mediciones,
    )


@router.get("/matriz/{job_id}/resultado", response_model=MatrizResult)
async def get_matriz_result(job_id: str):
    """Get radiance matrix and municipality mask. Returns 409 if job is not yet completed."""
    state = job_store.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Job not finished (status: {state.status}). Poll GET /matriz/{job_id}.",
        )
    if state.status == "failed":
        raise HTTPException(
            status_code=409,
            detail=f"Job failed: {state.error or 'Unknown error'}",
        )
    if not state.results:
        raise HTTPException(status_code=500, detail="No result data")
    return MatrizResult.model_validate(state.results[0])
