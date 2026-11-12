"""
Descarga las imágenes Black Marble (VJ146A2 por omisión) de 8 cuadrantes (H06V06..H09V06 y H06V07..H09V07)
para un día dado y las une en un único mosaico en blanco y negro (escala de grises).

Requiere NASA_API_TOKEN en el .env (token de LAADS DAAC / Earthdata) para poder
descargar los archivos HDF5 reales.

Uso:
    python scripts/mosaico_cuadrantes.py [--fecha DD-MM-YY] [--salida ruta.png]
"""
import argparse
import asyncio
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiohttp
import h5py

from ntl.core.config import find_image_path
from ntl.core.downloader import download_file_async, find_file_async
from ntl.core.utils import parse_date

# Columnas H06..H09 (de izquierda a derecha) x filas V06..V07 (de arriba a abajo,
# ya que v crece hacia el sur en la cuadrícula sinusoidal de VIIRS/MODIS).
COLUMNAS = ["h06", "h07", "h08", "h09"]
FILAS = ["v06", "v07"]
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp", "mosaico")


def radiancia_a_grises(matrix: np.ndarray, tope: float) -> np.ndarray:
    """Normaliza una matriz de radianza a 8 bits (0-255) usando un tope común a todo el mosaico."""
    clipped = np.clip(matrix, 0, tope)
    return (clipped / tope * 255).astype(np.uint8)


def es_hdf5_valido(path: str) -> bool:
    """Verifica que el archivo exista, no sea una página HTML y sea un HDF5 abrible."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            start = f.read(15)
            if b"<html" in start or b"<!DOCTYPE html" in start:
                return False
        with h5py.File(path, "r"):
            pass
        return True
    except Exception:
        return False


async def descargar_cuadrante(
    session: aiohttp.ClientSession, year: str, day: str, cuadrante: str, intentos: int = 3
) -> str | None:
    save_path = os.path.join(TEMP_DIR, f"{year}{day}_{cuadrante}.h5")

    if es_hdf5_valido(save_path):
        print(f"  [{cuadrante}] reutilizando archivo ya descargado en {save_path}")
        return save_path

    for intento in range(1, intentos + 1):
        try:
            url = await find_file_async(session, year, day, cuadrante)
            if not url:
                print(f"  [{cuadrante}] no se encontró archivo para {year}-{day}")
                return None
            downloaded = await download_file_async(session, url, save_path)
            if downloaded and es_hdf5_valido(downloaded):
                return downloaded
            print(f"  [{cuadrante}] descarga inválida o incompleta (intento {intento}/{intentos})")
        except Exception as e:
            print(f"  [{cuadrante}] error en intento {intento}/{intentos}: {e}")
        if intento < intentos:
            await asyncio.sleep(5)

    print(f"  [{cuadrante}] se omite tras {intentos} intentos fallidos")
    return None


def leer_matriz_radianza(h5_path: str) -> np.ndarray | None:
    if not es_hdf5_valido(h5_path):
        print(f"  {h5_path}: archivo no válido (revisa NASA_API_TOKEN)")
        return None
    with h5py.File(h5_path, "r") as hdf_file:
        radiance_path = find_image_path(hdf_file)
        image_matrix = hdf_file[radiance_path][()]
    return np.nan_to_num(image_matrix.astype(float), nan=0.0, posinf=0.0, neginf=0.0)


async def construir_mosaico(fecha_str: str, salida: str) -> str:
    year, day, date_obj = parse_date(fecha_str)
    os.makedirs(TEMP_DIR, exist_ok=True)

    cuadrantes = [f"{col}{fila}" for fila in FILAS for col in COLUMNAS]
    print(f"Descargando {len(cuadrantes)} cuadrantes para {date_obj} ({year}-{day})...")

    async with aiohttp.ClientSession() as session:
        rutas = {}
        for cuadrante in cuadrantes:
            print(f"- {cuadrante}")
            rutas[cuadrante] = await descargar_cuadrante(session, year, day, cuadrante)

    radiancias: dict[str, np.ndarray] = {}
    tile_shape = None
    for cuadrante, ruta in rutas.items():
        if ruta is None:
            continue
        matriz = leer_matriz_radianza(ruta)
        if matriz is not None:
            radiancias[cuadrante] = matriz
            tile_shape = matriz.shape

    for cuadrante, ruta in rutas.items():
        if ruta and os.path.exists(ruta):
            os.remove(ruta)

    if not radiancias:
        raise RuntimeError(
            "No se pudo obtener ninguna imagen real. Verifica NASA_API_TOKEN en .env."
        )

    if tile_shape is None:
        raise RuntimeError("No se determinó el tamaño de los tiles.")

    # Tope común (percentil 99 de todos los cuadrantes juntos) para que el brillo
    # sea comparable entre tiles y el mosaico se vea homogéneo, sin costuras.
    tope_global = float(np.percentile(np.concatenate([m.ravel() for m in radiancias.values()]), 99)) or 1.0
    print(f"Tope de radianza global (percentil 99): {tope_global:.2f}")

    tiles = {cuadrante: radiancia_a_grises(matriz, tope_global) for cuadrante, matriz in radiancias.items()}

    relleno = np.zeros(tile_shape, dtype=np.uint8)
    filas_imagenes = []
    for fila in FILAS:
        fila_tiles = [tiles.get(f"{col}{fila}", relleno) for col in COLUMNAS]
        filas_imagenes.append(np.hstack(fila_tiles))
    mosaico = np.vstack(filas_imagenes)

    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    Image.fromarray(mosaico, mode="L").save(salida)
    print(f"\nCuadrantes obtenidos: {len(tiles)}/{len(cuadrantes)}")
    print(f"Mosaico guardado en: {salida}")
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fecha", default="15-01-24", help="Fecha en formato DD-MM-YY (default 15-01-24)")
    parser.add_argument(
        "--salida",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "images",
            "mosaico_h06-09_v06-07.png",
        ),
        help="Ruta de salida de la imagen PNG",
    )
    args = parser.parse_args()
    asyncio.run(construir_mosaico(args.fecha, args.salida))


if __name__ == "__main__":
    main()
