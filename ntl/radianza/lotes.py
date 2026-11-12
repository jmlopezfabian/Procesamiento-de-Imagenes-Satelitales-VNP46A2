import asyncio
import errno
import glob
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable

import pandas as pd

from ..core.config import (MAX_FECHAS_CONCURRENTES, PIXELES_MUNICIPIOS,
                           TEMP_DIR, data_path, temp_path)
from ..core.errores import CredencialesInvalidas
from ..core.utils import normalize_municipio, parse_date, load_coord_data
from ..core.downloader import (buscar_archivo_async, bajar_archivo_async,
                               iniciar_corrida, terminar_corrida)
from ..core.productos import obtener_producto
from .extraccion import process_image_mosaico
from ..core.models import MedicionResultado

def chunk_list(lst, chunk_size):
    """Divide una lista en chunks del tamaño especificado"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def cleanup_temp_files(directorio=None):
    """Limpia archivos .h5 y .part residuales de un directorio (por omisión TEMP_DIR).

    Es una herramienta manual, y es amplia a propósito: borra TODOS los .h5 del
    directorio, también los de otras corridas en curso. Por eso `run` ya no la
    llama: usa un subdirectorio propio (TEMP_DIR/run_<uuid>) que borra entero al
    terminar, sin tocar nada ajeno.
    """
    temp_dir = str(directorio if directorio is not None else TEMP_DIR)
    if os.path.exists(temp_dir):
        residuos = (glob.glob(os.path.join(temp_dir, "*.h5"))
                    + glob.glob(os.path.join(temp_dir, "*.part")))
        for file_path in residuos:
            try:
                os.remove(file_path)
                print(f"Archivo residual eliminado: {file_path}")
            except Exception as e:
                print(f"Error eliminando archivo residual {file_path}: {e}")

def save_progress(df, municipio, chunk_number=None, producto=None):
    """Guarda el progreso actual en Parquet, de forma atómica.

    Con `producto` el nombre lo incluye (`{municipio}_{producto}_progress.parquet`):
    dos corridas de productos distintos en el mismo DATA_DIR no se pisan el
    checkpoint.

    Se escribe a un temporal y se renombra: si el proceso muere a mitad, el
    checkpoint anterior sigue entero en lugar de quedar un Parquet truncado.
    """
    filename = None
    try:
        base = f"{municipio}_{producto}" if producto else municipio
        if chunk_number is not None:
            filename = str(data_path(f"{base}_progress_chunk_{chunk_number}.parquet"))
        else:
            filename = str(data_path(f"{base}_progress.parquet"))

        temporal = f"{filename}.{uuid.uuid4().hex}.tmp"
        try:
            df.to_parquet(temporal, index=False)
            os.replace(temporal, filename)
        finally:
            if os.path.exists(temporal):
                os.remove(temporal)
        print(f"✅ Progreso guardado: {filename} ({len(df)} registros)")
        return filename
    except Exception as e:
        print(f"❌ Error guardando progreso: {e}")
        return None

class SatelliteImagesAsync:
    """
    Class for get the measures of the satellite images for multiple municipalities
    """
    
    def __init__(self, municipios, producto=None):
        """
        Inicializa con una lista de municipios
        
        Args:
            municipios: Lista de nombres de municipios o string único
            producto: nombre o `Producto` de Black Marble (VNP46A2, VJ146A2).
                None = el por omisión (NTL_PRODUCTO, o VJ146A2). Se valida aquí.
                Una instancia usa un solo producto: nunca se mezclan ni se
                elige uno por fecha (ver core.productos).
        """
        if isinstance(municipios, str):
            municipios = [municipios]
        
        self.producto = obtener_producto(producto)
        self.municipios = [normalize_municipio(m) for m in municipios]
        self.coord_data_dict = {}
        self.cache_h5_files = {}  # Cache para archivos H5 ya descargados
        # Mediciones que no se pudieron calcular por un fallo, no por falta de
        # datos. Una fila que no está no dice por qué no está; sin esta lista,
        # un municipio que revienta y una noche nublada dejan el mismo hueco.
        self.fallos = []
        # Descargas en vuelo por (año, día, cuadrante), compartidas entre los
        # municipios de una fecha, y directorio propio de la corrida en curso.
        # Los pone `run`; `get_measures_for_date` también funciona sin él.
        self._vuelo = {}
        self._dir_temp = None

        # Cargar datos de coordenadas para todos los municipios
        for municipio in self.municipios:
            self.coord_data_dict[municipio] = load_coord_data(municipio, PIXELES_MUNICIPIOS)
        
        print(f"✅ Inicializado con {len(self.municipios)} municipios: {', '.join(self.municipios)}")

    def _ruta_temp(self, nombre):
        """Ruta de un gránulo: en el directorio de la corrida si la hay."""
        if self._dir_temp is not None:
            return str(Path(self._dir_temp) / nombre)
        return str(temp_path(nombre))

    async def _bajar_h5(self, session, year, day, cuadrante, date_obj, cache_key):
        print(f"🔍 Buscando archivo H5 para: {year}-{day} ({cuadrante})")
        # Las variantes que lanzan `DescargaFallida` al agotar reintentos: aquí
        # None solo significa «no existe», y un fallo de red no puede pasar por
        # falta de datos. La excepción sube a `get_measures_for_date`, que la
        # anota en `fallos`.
        h5_url = await buscar_archivo_async(session, year, day, cuadrante, self.producto)
        if not h5_url:
            print(f"❌ No se encontró archivo H5 para: {year}-{day} ({cuadrante})")
            return None

        save_path = self._ruta_temp(f"{date_obj}_{self.producto.nombre}_{cuadrante}.h5")
        print(f"📥 Descargando: {h5_url} -> {save_path}")
        downloaded_path = await bajar_archivo_async(session, h5_url, save_path)

        if downloaded_path:
            self.cache_h5_files[cache_key] = downloaded_path
            print(f"✅ Archivo H5 descargado y cacheado: {cache_key}")
            return downloaded_path
        print(f"❌ El archivo H5 no existe en el servidor: {h5_url}")
        return None

    async def _download_and_cache_h5(self, session, year, day, cuadrante, date_obj):
        """
        Descarga un archivo H5 y lo cachea para reutilización.

        Single-flight por (año, día, cuadrante): los municipios de una fecha que
        comparten cuadrante esperan la misma descarga, y el resultado se
        memoriza también cuando es None. Sin eso, un cuadrante que no existe o
        que falla se buscaba y reintentaba una vez por municipio, cada una con
        sus propios reintentos contra el servidor.
        """
        cache_key = f"{year}_{day}_{cuadrante}"

        if cache_key in self.cache_h5_files:
            print(f"✅ Usando archivo H5 cacheado: {cache_key}")
            return self.cache_h5_files[cache_key]

        # Sin ningún await entre la consulta y el registro: dos llamadores no
        # pueden crear dos descargas de la misma clave.
        tarea = self._vuelo.get(cache_key)
        if tarea is None:
            tarea = asyncio.ensure_future(
                self._bajar_h5(session, year, day, cuadrante, date_obj, cache_key))
            self._vuelo[cache_key] = tarea
        # shield: que cancelar a un llamador no mate la descarga que otros esperan.
        return await asyncio.shield(tarea)

    def _borrar_cuadrante(self, cache_key, ruta):
        """Borra un HDF5 ya consumido y lo saca del cache, que si no apuntaría a la nada."""
        self.cache_h5_files.pop(cache_key, None)
        # También se olvida la descarga compartida (y con ella la caché negativa):
        # su resultado ya no es válido, y la fecha terminó.
        tarea = self._vuelo.pop(cache_key, None)
        if tarea is not None and not tarea.done():
            tarea.cancel()
        try:
            if ruta and os.path.exists(ruta):
                os.remove(ruta)
                print(f"Archivo eliminado: {ruta}")
        except Exception as e:
            print(f"Error eliminando archivo {ruta}: {e}")

    async def get_measures_for_date(self, session, date_str, estricto=False):
        """
        Medidas de todos los municipios en una fecha.

        Un municipio puede necesitar más de un cuadrante —la retícula se corta
        cada 10 grados, sin mirar fronteras administrativas— y un cuadrante
        suele servir a varios municipios. Así que no se agrupa por "el cuadrante
        del municipio": se cuenta cuántos municipios quedan por procesar de cada
        cuadrante y la imagen se borra en cuanto ese contador llega a cero. Cada
        archivo pesa cientos de megas; mantenerlos todos en disco hasta el final
        era la alternativa fácil y la que llena el volumen.

        Los cuadrantes de la fecha se descargan en paralelo (el limitador global
        decide cuántos bajan de verdad a la vez), y un fallo de red al buscarlos
        o bajarlos se anota en `fallos` igual que uno de procesamiento.
        `CredencialesInvalidas` no: aborta, porque todo lo demás fallará igual.
        """
        year, day, date_obj = parse_date(date_str)
        results = []

        # Fuera de la vigencia del producto no se pide nada a NASA: se anota en
        # `fallos` (no es falta de imagen, es una fecha que este producto no
        # cubre) y la corrida sigue con las demás fechas.
        motivo = self.producto.motivo_fuera_de_rango(date_obj)
        if motivo:
            print(f"⚠️ {date_obj}: {motivo}; no se pide a NASA")
            for municipio in self.municipios:
                self.fallos.append({
                    "municipio": municipio,
                    "fecha": str(date_obj),
                    "cuadrantes": list(self.coord_data_dict[municipio].cuadrantes),
                    "tipo": "ProductoFueraDeRango",
                    "mensaje": motivo,
                })
            if estricto:
                raise ValueError(motivo)
            return results

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

        liberados = set()
        # Arranca ya todas las descargas de la fecha: el primer municipio no
        # tiene por qué esperar a los cuadrantes que solo usa el último.
        precarga = [
            asyncio.ensure_future(
                self._download_and_cache_h5(session, year, day, cuadrante, date_obj))
            for cuadrante in dict.fromkeys(
                c for m in orden for c in cuadrantes_por_municipio[m])
        ]

        try:
            for municipio in orden:
                cuadrantes = cuadrantes_por_municipio[municipio]
                try:
                    bajadas = await asyncio.gather(*[
                        self._download_and_cache_h5(session, year, day, c, date_obj)
                        for c in cuadrantes
                    ])
                    rutas = dict(zip(cuadrantes, bajadas))

                    # h5py/numpy son CPU y disco puros: en el loop congelaban
                    # también las descargas en curso.
                    datos = await asyncio.to_thread(
                        process_image_mosaico,
                        rutas,
                        self.coord_data_dict[municipio].piezas,
                        date_obj,
                        municipio,
                        # El borrado lo decide el contador, no el municipio: la
                        # misma imagen le sirve al siguiente.
                        delete_files=False,
                        # El archivo declara su producto (ShortName): si no es
                        # el de esta instancia, se rechaza en vez de mezclarlo.
                        producto=self.producto.nombre,
                    )
                    if datos:
                        results.append(datos.model_dump())
                        print(f"✅ Procesado: {municipio} - {date_obj}")
                    else:
                        print(f"⚠️ Sin datos para: {municipio} - {date_obj}")
                except CredencialesInvalidas:
                    raise
                except Exception as e:
                    if isinstance(e, OSError) and e.errno == errno.ENOSPC:
                        # Disco lleno: cada fecha siguiente fallaría igual.
                        raise
                    # Se anota antes de seguir: la corrida sigue, pero la serie
                    # queda con un hueco y tiene que quedar dicho de quién y cuándo.
                    self.fallos.append({
                        "municipio": municipio,
                        "fecha": str(date_obj),
                        "cuadrantes": list(cuadrantes),
                        "tipo": type(e).__name__,
                        "mensaje": str(e),
                    })
                    print(f"❌ Error procesando {municipio} para {date_obj}: "
                          f"{type(e).__name__}: {e}")
                    if estricto:
                        raise

                for cuadrante in cuadrantes:
                    pendientes[cuadrante] -= 1
                    if pendientes[cuadrante] == 0:
                        clave = f"{year}_{day}_{cuadrante}"
                        self._borrar_cuadrante(clave, self.cache_h5_files.get(clave))
                        liberados.add(cuadrante)
        finally:
            # Si se aborta a medias, nada de esta fecha puede quedar bajando ni
            # ocupando disco.
            for cuadrante in pendientes:
                if cuadrante not in liberados:
                    clave = f"{year}_{day}_{cuadrante}"
                    self._borrar_cuadrante(clave, self.cache_h5_files.get(clave))
            for t in precarga:
                t.cancel()
            await asyncio.gather(*precarga, return_exceptions=True)

        return results

    @staticmethod
    def _sin_repetidas(fechas):
        """Quita las fechas repetidas (mismo día, aunque esté escrito distinto) conservando el orden."""
        vistas, unicas = set(), []
        for f in fechas:
            try:
                clave = parse_date(f)[2]
            except (ValueError, TypeError):
                clave = f
            if clave not in vistas:
                vistas.add(clave)
                unicas.append(f)
        return unicas

    async def run(self, fechas, chunks=None, save_progress_enabled=True,
                  on_progress: Callable[[str], None] | None = None,
                  max_concurrentes: int | None = None, estricto: bool = False):
        """
        Procesa una lista de fechas.

        `max_concurrentes` acota cuántas fechas se procesan a la vez y con ello
        el pico de disco (ver MAX_FECHAS_CONCURRENTES en core.config). Es un
        pool de workers de tamaño constante: apenas uno termina toma la
        siguiente fecha, sin barreras. Cuántas descargas bajan de verdad a la
        vez lo decide aparte el limitador global (core.limitador).

        `chunks` es cada cuántas fechas completadas se reescribe el checkpoint,
        un único archivo (`multi_municipio_{producto}_progress.parquet`) que se reemplaza
        de forma atómica. Si la corrida aborta se guarda una última vez.

        `self.fallos` se reinicia al empezar: describe esta corrida, no la suma
        de las anteriores de la instancia.

        Las fechas repetidas se procesan una sola vez. Una medición que falla no
        detiene la corrida: se anota en `self.fallos` y se resume al terminar.
        `estricto=True` aborta en el primer fallo, que es lo que quiere quien
        reconstruye una serie desde cero; quien barre diez años prefiere
        terminar y mirar la lista. `CredencialesInvalidas` siempre aborta.
        """
        import aiohttp
        fechas = self._sin_repetidas(fechas)
        total_fechas = len(fechas)
        n_workers = max(1, max_concurrentes if max_concurrentes is not None
                        else MAX_FECHAS_CONCURRENTES)
        por_fecha = {}  # índice de la fecha -> filas; se concatena en orden de entrada
        completadas = 0
        candado_checkpoint = asyncio.Lock()
        cola = asyncio.Queue()
        for item in enumerate(fechas):
            cola.put_nowait(item)

        # Directorio propio: lo que baje esta corrida es lo único que se borra
        # al final, sin tocar los .h5 de otra corrida que comparta TEMP_DIR.
        self._vuelo = {}
        self.cache_h5_files = {}
        self.fallos = []
        self._dir_temp = Path(TEMP_DIR) / f"run_{uuid.uuid4().hex}"
        self._dir_temp.mkdir(parents=True, exist_ok=True)

        def _filas():
            return [r for i in sorted(por_fecha) for r in por_fecha[i]]

        async def _checkpoint():
            async with candado_checkpoint:
                filas = _filas()
                if filas:
                    # Armar el DataFrame y escribir el Parquet bloquea: en un hilo.
                    await asyncio.to_thread(
                        lambda: save_progress(pd.DataFrame(filas), "multi_municipio",
                                              producto=self.producto.nombre))

        async def worker():
            nonlocal completadas
            while True:
                try:
                    idx, fecha = cola.get_nowait()
                except asyncio.QueueEmpty:
                    return
                datos = await self.get_measures_for_date(session, fecha, estricto)
                por_fecha[idx] = datos or []
                completadas += 1
                if on_progress is not None:
                    on_progress(f"{completadas}/{total_fechas} fechas")
                if (chunks and save_progress_enabled and completadas % chunks == 0
                        and completadas < total_fechas):
                    await _checkpoint()

        workers = []
        iniciar_corrida()
        try:
            async with aiohttp.ClientSession() as session:
                # Tareas explícitas: si una falla o se cancela `run`, se cancelan
                # y se esperan aquí, antes de cerrar la sesión. Con as_completed
                # quedaban huérfanas usando una sesión cerrada.
                workers = [asyncio.ensure_future(worker())
                           for _ in range(min(n_workers, total_fechas))]
                try:
                    await asyncio.gather(*workers)
                finally:
                    for w in workers:
                        w.cancel()
                    await asyncio.gather(*workers, return_exceptions=True)
                    vuelo = list(self._vuelo.values())
                    for t in vuelo:
                        t.cancel()
                    await asyncio.gather(*vuelo, return_exceptions=True)
        except Exception as e:
            print(f"❌ Error durante el procesamiento: {e}")
            # Un único archivo de progreso, el mismo de los checkpoints.
            if por_fecha and save_progress_enabled:
                await _checkpoint()
            raise
        finally:
            shutil.rmtree(self._dir_temp, ignore_errors=True)
            self._dir_temp = None
            self._vuelo = {}
            # Los listados recordados son fotos de un momento: no se llevan a la
            # siguiente corrida de un proceso largo (ni se retienen sin fin).
            terminar_corrida()
            self._resumir_fallos(len(_filas()), save_progress_enabled)

        return pd.DataFrame(_filas())

    def _resumir_fallos(self, n_resultados, guardar=True):
        """
        Deja constancia de los huecos que no son huecos de datos.

        Un resumen impreso en una terminal que luego se cierra no es evidencia,
        así que además se escribe al lado de los resultados.
        """
        if not self.fallos:
            return
        total = n_resultados + len(self.fallos)
        porcentaje = 100 * len(self.fallos) / total if total else 0.0
        tipos = {}
        for f in self.fallos:
            tipos[f["tipo"]] = tipos.get(f["tipo"], 0) + 1
        detalle = ", ".join(f"{n} {t}" for t, n in sorted(tipos.items()))
        print(f"⚠️ {len(self.fallos)} de {total} mediciones fallaron "
              f"({porcentaje:.2f}%): {detalle}. Faltan esas filas de la serie y "
              f"no es porque no hubiera imagen. El detalle está en `.fallos`.")
        if guardar:
            try:
                ruta = data_path(f"fallos_{self.producto.nombre}.parquet")
                pd.DataFrame(self.fallos).to_parquet(ruta, index=False)
                print(f"   Detalle guardado en {ruta}")
            except Exception as e:
                print(f"   No se pudo guardar el detalle de fallos: {e}")