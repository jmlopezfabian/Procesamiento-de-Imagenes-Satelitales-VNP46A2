"""In-memory job store and background job execution for satellite processing."""
import asyncio
import os
from datetime import date, datetime
from typing import Literal

from ntl.core.config import PIXELES_MUNICIPIOS, temp_path
from ntl.core.downloader import download_file_async, find_file_async
from ntl.radianza.extraccion import extract_radiance_matrix_mosaico
from ntl.radianza.lotes import SatelliteImagesAsync
from ntl.core.utils import load_coord_data, normalize_municipio, parse_date


class JobState:
    """Mutable state for a single job."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status: Literal["pending", "running", "completed", "failed"] = "pending"
        self.progress: str | None = None
        self.created_at = datetime.utcnow()
        self.finished_at: datetime | None = None
        self.error: str | None = None
        self.results: list[dict] = []
        self.total_results: int = 0
        # Mediciones que fallaron. Un job "completed" con filas de menos y sin
        # este campo dice que todo fue bien cuando no lo fue.
        self.fallos: list[dict] = []
        self.task: asyncio.Task | None = None


class JobStore:
    """In-memory store for job states. Single instance used by the API."""

    def __init__(self):
        self._jobs: dict[str, JobState] = {}

    def create(self, job_id: str) -> JobState:
        state = JobState(job_id)
        self._jobs[job_id] = state
        return state

    def get(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def set_task(self, job_id: str, task: asyncio.Task) -> None:
        state = self._jobs.get(job_id)
        if state:
            state.task = task

    def remove(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)


job_store = JobStore()


async def run_job(
    job_id: str,
    municipios: list[str],
    fechas: list[str],
    chunks: int | None,
) -> None:
    """
    Run satellite processing in the background. Updates the job state in job_store.
    """
    state = job_store.get(job_id)
    if not state:
        return
    state.status = "running"
    state.progress = "0/" + str(len(fechas)) + " fechas"

    def on_progress(progress: str) -> None:
        if state:
            state.progress = progress

    try:
        sat = SatelliteImagesAsync(municipios)
        df = await sat.run(
            fechas,
            chunks=chunks,
            save_progress_enabled=False,
            on_progress=on_progress,
        )
        state.results = df.to_dict(orient="records") if not df.empty else []
        state.fallos = sat.fallos
        state.status = "completed"
        state.total_results = len(state.results)
    except asyncio.CancelledError:
        state.status = "failed"
        state.error = "Job cancelled"
    except Exception as e:
        state.status = "failed"
        state.error = str(e)
    finally:
        state.finished_at = datetime.utcnow()


async def run_matriz_job(job_id: str, municipio: str, fecha: date) -> None:
    """
    Run matriz extraction in the background. Downloads HDF5, extracts radiance
    submatrix and municipality mask, stores result in job_store.
    """
    state = job_store.get(job_id)
    if not state:
        return
    state.status = "running"
    state.progress = "Descargando imagen..."

    try:
        municipio_norm = normalize_municipio(municipio)
        coord_data = load_coord_data(municipio_norm, PIXELES_MUNICIPIOS)
        date_str = fecha.strftime("%d-%m-%y")
        year, day, date_obj = parse_date(date_str)
        # Un municipio puede ocupar hasta cuatro cuadrantes; se descargan todos
        # y el recorte se compone en el marco del cuadrante de referencia.
        cuadrantes = coord_data.cuadrantes

        import aiohttp

        rutas: dict[str, str] = {}
        async with aiohttp.ClientSession() as session:
            for cuadrante in cuadrantes:
                h5_url = await find_file_async(session, year, day, cuadrante)
                if not h5_url:
                    state.status = "failed"
                    state.error = f"No se encontró archivo HDF5 para {year}-{day} ({cuadrante})"
                    return

                save_path = str(temp_path(f"{date_obj}_{cuadrante}_matriz.h5"))
                downloaded_path = await download_file_async(session, h5_url, save_path)
                if not downloaded_path:
                    state.status = "failed"
                    state.error = f"Error descargando archivo HDF5 de {cuadrante}"
                    return
                rutas[cuadrante] = downloaded_path

        state.progress = "Extrayendo matrices..."
        result = extract_radiance_matrix_mosaico(
            rutas,
            coord_data.piezas,
            date_obj,
            municipio_norm,
        )

        for ruta in rutas.values():
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
            except OSError:
                pass

        if result is None:
            state.status = "failed"
            state.error = "No se pudo extraer la matriz de radianza"
            return

        result["job_id"] = job_id
        state.results = [result]
        state.status = "completed"
        state.total_results = 1

    except asyncio.CancelledError:
        state.status = "failed"
        state.error = "Job cancelled"
    except Exception as e:
        state.status = "failed"
        state.error = str(e)
    finally:
        state.finished_at = datetime.utcnow()
