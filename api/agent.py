"""PydanticAI agent with tools for Black Marble (VNP46A2) data queries."""
import asyncio
import json
import os
import uuid
from datetime import date, timedelta

import numpy as np
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from ntl.core.config import PIXELES_MUNICIPIOS
from ntl.core.utils import normalize_municipio

from .job_manager import job_store, run_job, run_matriz_job

load_dotenv()

# Store last tool results for the chat endpoint to include in the response
_last_tool_results: dict = {"heatmap_data": None, "mediciones": None}


def _get_available_municipios() -> list[str]:
    """Load municipality names from config JSON path."""
    with open(PIXELES_MUNICIPIOS, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys())


def _to_json_serializable(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_json_serializable(obj.tolist())
    if isinstance(obj, list):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    return obj


def _build_fechas(fecha_inicio: date, fecha_fin: date) -> list[str]:
    """Build list of date strings dd-mm-yy from inicio to fin (inclusive)."""
    fechas = []
    d = fecha_inicio
    while d <= fecha_fin:
        fechas.append(d.strftime("%d-%m-%y"))
        d += timedelta(days=1)
    return fechas


def _create_agent() -> Agent:
    """Create and return the PydanticAI agent with tools."""
    api_key = os.environ.get("GOOGLE_AI_TOKEN") or os.environ.get("GOOGLE_API_KEY")
    provider = GoogleProvider(api_key=api_key) if api_key else GoogleProvider()
    model = GoogleModel("gemini-3.1-flash-lite-preview", provider=provider)

    agent = Agent(
        model,
        instructions=(
            "Eres un asistente experto en datos de luminosidad nocturna VNP46A2. "
            "Ayudas a consultar radianza por municipio y fecha. "
            "Cuando el usuario pregunte por un municipio y fecha, usa las herramientas para obtener datos "
            "y responde con un análisis breve. "
            "Si obtienes una matriz de radianza, describe los valores y avísale al usuario que se mostrará el heatmap."
        ),
        deps_type=type(None),
    )

    @agent.tool_plain
    async def list_municipios() -> list[str]:
        """Lista los nombres de municipios disponibles para consultar."""
        print("[Agent] Tool called: list_municipios()")
        municipios = _get_available_municipios()
        print(f"[Agent] list_municipios -> returning {len(municipios)} municipios")
        return municipios

    @agent.tool_plain
    async def get_mediciones(
        municipio: str,
        fecha_inicio: str,
        fecha_fin: str,
    ) -> dict:
        """
        Obtiene mediciones de radianza para un municipio en un rango de fechas.
        municipio: nombre del municipio (ej. iztapalapa)
        fecha_inicio: fecha inicio YYYY-MM-DD
        fecha_fin: fecha fin YYYY-MM-DD
        """
        global _last_tool_results
        print(f"[Agent] Tool called: get_mediciones(municipio={municipio!r}, fecha_inicio={fecha_inicio}, fecha_fin={fecha_fin})")
        _last_tool_results["heatmap_data"] = None
        _last_tool_results["mediciones"] = None

        available = _get_available_municipios()
        normalized = normalize_municipio(municipio)
        if normalized not in available:
            return {"error": f"Municipio '{municipio}' no disponible. Usa list_municipios para ver la lista."}

        d_inicio = date.fromisoformat(fecha_inicio)
        d_fin = date.fromisoformat(fecha_fin)
        if d_fin < d_inicio:
            return {"error": "fecha_fin debe ser >= fecha_inicio"}

        fechas = _build_fechas(d_inicio, d_fin)
        if not fechas:
            return {"error": "No hay fechas en el rango"}

        job_id = str(uuid.uuid4())
        job_store.create(job_id)
        task = asyncio.create_task(
            run_job(job_id, [normalized], fechas, None)
        )
        job_store.set_task(job_id, task)
        print(f"[Agent] get_mediciones: job {job_id[:8]}... running, waiting...")
        await task

        state = job_store.get(job_id)
        if not state:
            print("[Agent] get_mediciones: job not found")
            return {"error": "Job no encontrado"}
        if state.status == "failed":
            print(f"[Agent] get_mediciones: job failed - {state.error}")
            return {"error": state.error or "Job falló"}
        if state.status != "completed":
            print(f"[Agent] get_mediciones: job status={state.status}")
            return {"error": f"Job en estado {state.status}"}

        records = [r.copy() for r in state.results]
        print(f"[Agent] get_mediciones -> completed, {len(records)} records")
        for r in records:
            if "Fecha" in r and hasattr(r["Fecha"], "isoformat"):
                r["Fecha"] = r["Fecha"].isoformat()
        records = _to_json_serializable(records)
        _last_tool_results["mediciones"] = records
        return {"count": len(records), "mediciones": records}

    @agent.tool_plain
    async def get_radiance_matrix(municipio: str, fecha: str) -> dict:
        """
        Obtiene la matriz de radianza y máscara del municipio para una fecha específica.
        municipio: nombre del municipio (ej. iztapalapa)
        fecha: fecha YYYY-MM-DD
        """
        global _last_tool_results
        print(f"[Agent] Tool called: get_radiance_matrix(municipio={municipio!r}, fecha={fecha})")
        _last_tool_results["heatmap_data"] = None
        _last_tool_results["mediciones"] = None

        available = _get_available_municipios()
        normalized = normalize_municipio(municipio)
        if normalized not in available:
            print(f"[Agent] get_radiance_matrix: municipio '{municipio}' not available")
            return {"error": f"Municipio '{municipio}' no disponible. Usa list_municipios para ver la lista."}

        d = date.fromisoformat(fecha)

        job_id = str(uuid.uuid4())
        job_store.create(job_id)
        task = asyncio.create_task(run_matriz_job(job_id, municipio, d))
        job_store.set_task(job_id, task)
        print(f"[Agent] get_radiance_matrix: job {job_id[:8]}... running, waiting...")
        await task

        state = job_store.get(job_id)
        if not state:
            print("[Agent] get_radiance_matrix: job not found")
            return {"error": "Job no encontrado"}
        if state.status == "failed":
            print(f"[Agent] get_radiance_matrix: job failed - {state.error}")
            return {"error": state.error or "Job falló"}
        if state.status != "completed" or not state.results:
            print(f"[Agent] get_radiance_matrix: job status={state.status}")
            return {"error": f"Job en estado {state.status}"}

        result = state.results[0]
        print(f"[Agent] get_radiance_matrix -> completed, matrix {result.get('rows',0)}x{result.get('cols',0)}")
        data = {
            "municipio": result.get("municipio", municipio),
            "fecha": result.get("fecha"),
            "rows": int(result.get("rows", 0)),
            "cols": int(result.get("cols", 0)),
            "radiance_matrix": _to_json_serializable(result.get("radiance_matrix", [])),
            "municipality_mask": _to_json_serializable(result.get("municipality_mask", [])),
            "municipality_coverage": _to_json_serializable(result.get("municipality_coverage", [])),
        }
        if hasattr(data["fecha"], "isoformat"):
            data["fecha"] = data["fecha"].isoformat()
        _last_tool_results["heatmap_data"] = data
        return data

    return agent


_agent_instance: Agent | None = None


def get_agent() -> Agent:
    """Return singleton agent instance."""
    global _agent_instance
    if _agent_instance is None:
        print("[Agent] Creating agent (first call)...")
        _agent_instance = _create_agent()
        print(f"[Agent] Agent ready (model: {_agent_instance.model.model_name})")
    return _agent_instance


def get_last_tool_results() -> tuple[dict | None, list | None]:
    """Return and clear last heatmap_data and mediciones from tools."""
    global _last_tool_results
    heatmap = _last_tool_results.get("heatmap_data")
    mediciones = _last_tool_results.get("mediciones")
    _last_tool_results["heatmap_data"] = None
    _last_tool_results["mediciones"] = None
    return heatmap, mediciones
