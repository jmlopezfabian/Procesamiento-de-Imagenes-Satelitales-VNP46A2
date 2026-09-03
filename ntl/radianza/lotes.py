import asyncio
import pandas as pd
import os
import glob
from typing import Callable

from ..core.config import (MAX_FECHAS_CONCURRENTES, PIXELES_MUNICIPIOS,
                           TEMP_DIR, data_path, temp_path)
from ..core.utils import normalize_municipio, parse_date, load_coord_data
from ..core.downloader import find_file_async, download_file_async
from .extraccion import process_image_mosaico
from ..core.models import MedicionResultado

def chunk_list(lst, chunk_size):
    """Divide una lista en chunks del tamaño especificado"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def cleanup_temp_files():
    """Limpia archivos .h5 residuales en el directorio temp.

    Usa la ruta absoluta TEMP_DIR (configurable con NTL_TEMP_DIR); antes
    era "../temp" relativo al cwd, asi que podia limpiar un directorio
    distinto del que se habia usado para descargar.
    """
    temp_dir = str(TEMP_DIR)
    if os.path.exists(temp_dir):
        h5_files = glob.glob(os.path.join(temp_dir, "*.h5"))
        for file_path in h5_files:
            try:
                os.remove(file_path)
                print(f"Archivo residual eliminado: {file_path}")
            except Exception as e:
                print(f"Error eliminando archivo residual {file_path}: {e}")

def save_progress(df, municipio, chunk_number=None):
    """Guarda el progreso actual en Parquet"""
    try:
        # Nombre del archivo con información del chunk
        if chunk_number is not None:
            filename = str(data_path(f"{municipio}_progress_chunk_{chunk_number}.parquet"))
        else:
            filename = str(data_path(f"{municipio}_progress.parquet"))
        
        # Guardar Parquet
        df.to_parquet(filename, index=False)
        print(f"✅ Progreso guardado: {filename} ({len(df)} registros)")
        return filename
    except Exception as e:
        print(f"❌ Error guardando progreso: {e}")
        return None

async def process_chunks(satellite_instance, fechas, chunks, session, municipio):
    """Procesa las fechas en chunks de forma asíncrona con guardado progresivo"""
    results = []
    fechas_chunks = chunk_list(fechas, chunks)
    
    for i, chunk_fechas in enumerate(fechas_chunks):
        print(f"Procesando chunk {i+1}/{len(fechas_chunks)} con {len(chunk_fechas)} fechas")
        
        try:
            # Procesar el chunk actual de forma asíncrona
            tasks = [satellite_instance.get_measures(session, f) for f in chunk_fechas]
            chunk_results = []
            
            for result in asyncio.as_completed(tasks):
                datos = await result
                if datos:
                    chunk_results.append(datos.model_dump())
            
            # Agregar resultados del chunk actual
            results.extend(chunk_results)
            print(f"Chunk {i+1} completado. Resultados obtenidos: {len(chunk_results)}")
            
            # Guardar progreso después de cada chunk
            if chunk_results:
                temp_df = pd.DataFrame(results)
                save_progress(temp_df, municipio, i+1)
            
        except Exception as e:
            print(f"❌ Error procesando chunk {i+1}: {e}")
            # Guardar progreso hasta el momento en caso de error
            if results:
                temp_df = pd.DataFrame(results)
                save_progress(temp_df, municipio, f"error_chunk_{i+1}")
            raise e
    
    return results

class SatelliteImagesAsync:
    """
    Class for get the measures of the satellite images for multiple municipalities
    """
    
    def __init__(self, municipios):
        """
        Inicializa con una lista de municipios
        
        Args:
            municipios: Lista de nombres de municipios o string único
        """
        if isinstance(municipios, str):
            municipios = [municipios]
        
        self.municipios = [normalize_municipio(m) for m in municipios]
        self.coord_data_dict = {}
        self.cache_h5_files = {}  # Cache para archivos H5 ya descargados
        
        # Cargar datos de coordenadas para todos los municipios
        for municipio in self.municipios:
            self.coord_data_dict[municipio] = load_coord_data(municipio, PIXELES_MUNICIPIOS)
        
        print(f"✅ Inicializado con {len(self.municipios)} municipios: {', '.join(self.municipios)}")

    async def _download_and_cache_h5(self, session, year, day, cuadrante, date_obj):
        """Descarga un archivo H5 y lo cachea para reutilización"""
        cache_key = f"{year}_{day}_{cuadrante}"
        
        if cache_key in self.cache_h5_files:
            print(f"✅ Usando archivo H5 cacheado: {cache_key}")
            return self.cache_h5_files[cache_key]
        
        # Buscar y descargar el archivo
        print(f"🔍 Buscando archivo H5 para: {year}-{day} ({cuadrante})")
        h5_url = await find_file_async(session, year, day, cuadrante)
        if not h5_url:
            print(f"❌ No se encontró archivo H5 para: {year}-{day} ({cuadrante})")
            return None
            
        save_path = str(temp_path(f"{date_obj}_{cuadrante}.h5"))
        print(f"📥 Descargando: {h5_url} -> {save_path}")
        downloaded_path = await download_file_async(session, h5_url, save_path)
        
        if downloaded_path:
            self.cache_h5_files[cache_key] = downloaded_path
            print(f"✅ Archivo H5 descargado y cacheado: {cache_key}")
            return downloaded_path
        else:
            print(f"❌ Error descargando archivo H5: {h5_url}")
        
        return None

    def _borrar_cuadrante(self, cache_key, ruta):
        """Borra un HDF5 ya consumido y lo saca del cache, que si no apuntaría a la nada."""
        self.cache_h5_files.pop(cache_key, None)
        try:
            if ruta and os.path.exists(ruta):
                os.remove(ruta)
                print(f"Archivo eliminado: {ruta}")
        except Exception as e:
            print(f"Error eliminando archivo {ruta}: {e}")

    async def get_measures_for_date(self, session, date_str):
        """
        Medidas de todos los municipios en una fecha.

        Un municipio puede necesitar más de un cuadrante —la retícula se corta
        cada 10 grados, sin mirar fronteras administrativas— y un cuadrante
        suele servir a varios municipios. Así que no se agrupa por "el cuadrante
        del municipio": se cuenta cuántos municipios quedan por procesar de cada
        cuadrante y la imagen se borra en cuanto ese contador llega a cero. Cada
        archivo pesa cientos de megas; mantenerlos todos en disco hasta el final
        era la alternativa fácil y la que llena el volumen.
        """
        year, day, date_obj = parse_date(date_str)
        results = []

        cuadrantes_por_municipio = {
            municipio: self.coord_data_dict[municipio].cuadrantes
            for municipio in self.municipios
        }
        pendientes = {}
        for cuadrantes in cuadrantes_por_municipio.values():
            for cuadrante in cuadrantes:
                pendientes[cuadrante] = pendientes.get(cuadrante, 0) + 1

        # Procesar juntos los municipios que comparten cuadrantes hace que los
        # contadores lleguen a cero pronto y el disco no acumule imágenes.
        orden = sorted(self.municipios, key=lambda m: cuadrantes_por_municipio[m])

        for municipio in orden:
            cuadrantes = cuadrantes_por_municipio[municipio]
            rutas = {}
            for cuadrante in cuadrantes:
                rutas[cuadrante] = await self._download_and_cache_h5(
                    session, year, day, cuadrante, date_obj
                )

            try:
                datos = process_image_mosaico(
                    rutas,
                    self.coord_data_dict[municipio].piezas,
                    date_obj,
                    municipio,
                    # El borrado lo decide el contador, no el municipio: la
                    # misma imagen le sirve al siguiente.
                    delete_files=False,
                )
                if datos:
                    results.append(datos.model_dump())
                    print(f"✅ Procesado: {municipio} - {date_obj}")
                else:
                    print(f"⚠️ Sin datos para: {municipio} - {date_obj}")
            except Exception as e:
                print(f"❌ Error procesando {municipio} para {date_obj}: {e}")

            for cuadrante in cuadrantes:
                pendientes[cuadrante] -= 1
                if pendientes[cuadrante] == 0:
                    self._borrar_cuadrante(f"{year}_{day}_{cuadrante}", rutas.get(cuadrante))

        return results

    async def _medir_fecha_limitada(self, session, fecha, limite):
        """
        `get_measures_for_date` con un permiso del semáforo tomado.

        El límite va por fecha y no por descarga a propósito: una fecha retiene
        sus gránulos hasta que termina su último municipio —el contador de
        referencias los libera al final—, así que acotar las descargas
        simultáneas no acotaría cuántos archivos hay residentes en disco, que es
        lo que llena el volumen.

        Es un envoltorio en vez de un `async with` dentro de
        `get_measures_for_date` para que esa siga siendo llamable tal cual, sin
        tener que fabricarle un semáforo.
        """
        async with limite:
            return await self.get_measures_for_date(session, fecha)

    async def run(self, fechas, chunks=None, save_progress_enabled=True,
                  on_progress: Callable[[str], None] | None = None,
                  max_concurrentes: int | None = None):
        """
        Procesa una lista de fechas.

        `max_concurrentes` acota cuántas fechas se procesan a la vez y con ello
        el pico de disco (ver MAX_FECHAS_CONCURRENTES en core.config). Es
        independiente de `chunks`, que decide cada cuántas fechas se guarda
        progreso: antes acotar el disco obligaba a pedir checkpoints que quizá
        no se querían.
        """
        results = []
        import aiohttp
        total_fechas = len(fechas)
        completed_count = 0
        # El semáforo se crea aquí y no en __init__ porque __init__ es síncrono
        # y puede ejecutarse sin un loop corriendo.
        limite = asyncio.Semaphore(
            max_concurrentes if max_concurrentes is not None else MAX_FECHAS_CONCURRENTES
        )

        def _report_progress():
            nonlocal completed_count
            completed_count += 1
            if on_progress is not None:
                on_progress(f"{completed_count}/{total_fechas} fechas")

        try:
            async with aiohttp.ClientSession() as session:
                if chunks is None:
                    # Procesamiento original: todas las fechas de forma asíncrona
                    tasks = [self._medir_fecha_limitada(session, f, limite) for f in fechas]
                    for result in asyncio.as_completed(tasks):
                        datos_list = await result
                        if datos_list:
                            results.extend(datos_list)
                        _report_progress()
                else:
                    # Procesamiento por chunks
                    fechas_chunks = chunk_list(fechas, chunks)
                    
                    for i, chunk_fechas in enumerate(fechas_chunks):
                        print(f"Procesando chunk {i+1}/{len(fechas_chunks)} con {len(chunk_fechas)} fechas")
                        
                        try:
                            # Procesar el chunk actual de forma asíncrona
                            tasks = [self._medir_fecha_limitada(session, f, limite) for f in chunk_fechas]
                            chunk_results = []
                            
                            for result in asyncio.as_completed(tasks):
                                datos_list = await result
                                if datos_list:
                                    chunk_results.extend(datos_list)
                            
                            # Agregar resultados del chunk actual
                            results.extend(chunk_results)
                            for _ in chunk_fechas:
                                _report_progress()
                            print(f"Chunk {i+1} completado. Resultados obtenidos: {len(chunk_results)}")
                            
                            # Guardar progreso después de cada chunk
                            if chunk_results and save_progress_enabled:
                                temp_df = pd.DataFrame(results)
                                save_progress(temp_df, "multi_municipio", i+1)
                            
                        except Exception as e:
                            print(f"❌ Error procesando chunk {i+1}: {e}")
                            # Guardar progreso hasta el momento en caso de error
                            if results and save_progress_enabled:
                                temp_df = pd.DataFrame(results)
                                save_progress(temp_df, "error_chunk", i+1)
                            raise e
        except Exception as e:
            print(f"❌ Error durante el procesamiento: {e}")
            # Guardar progreso hasta el momento en caso de error
            if results and save_progress_enabled:
                temp_df = pd.DataFrame(results)
                save_progress(temp_df, "error_final", None)
            raise e
        finally:
            # Limpiar archivos residuales al final
            cleanup_temp_files()
        
        return pd.DataFrame(results)