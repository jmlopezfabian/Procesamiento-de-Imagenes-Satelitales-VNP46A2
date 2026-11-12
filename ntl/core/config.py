import os
import tempfile
from dotenv import load_dotenv
from importlib import resources
from pathlib import Path

load_dotenv()


def _primera_definida(*nombres: str) -> str | None:
    """
    Primer valor no vacío de una lista de variables de entorno.

    Las variables llevaban el prefijo VNP46A1_, que era el nombre anterior del
    paquete. Se leen los dos nombres para no romper un despliegue que ya tenga
    el antiguo configurado; el nuevo tiene precedencia.
    """
    for nombre in nombres:
        valor = os.getenv(nombre)
        if valor:
            return valor
    return None


# Producto de Black Marble del que se lee la radianza.
#
# Los productos soportados viven en `ntl.core.productos` (VNP46A2 de Suomi NPP,
# que termina el 2026-11-02, y VJ146A2 de NOAA-20, vigente). Ambos son versiones
# corregidas por reflectancia bidireccional lunar y atmósfera, con
# Mandatory_Quality_Flag por píxel. Por omisión se usa VJ146A2; NTL_PRODUCTO lo
# cambia. `PRODUCTO` se conserva por compatibilidad como el NOMBRE del producto
# por omisión al importar; el código interno resuelve un `Producto` con
# `productos.obtener_producto` y construye las URLs desde él.
#
# VNP46A1 sigue explícitamente fuera. Entrega radianza cruda al sensor, sin
# corregir y sin ninguna bandera que permita descartar una observación
# inservible: una noche completamente nublada produce un valor plausible que no
# es luz del suelo. Dejarlo disponible como opción invitaba a producir series
# que parecen válidas y no lo son.
PRODUCTO = _primera_definida("NTL_PRODUCTO") or "VJ146A2"

# API JSON de LAADS con el listado de archivos de un día. Es módulo-nivel para
# que las pruebas la apunten a un servidor local.
LAADS_API_BASE = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData"

# DEPRECADO: plantilla del listado HTML antiguo, solo por si algo externo la
# importa. Nada de ntl la usa: las URLs salen de `Producto.url_listado`.
BASE_URL = (
    "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/"
    f"{PRODUCTO}/{{year}}/{{day}}/"
)

# Rutas del dataset de radianza, en orden de preferencia: la colección publica
# el grid bajo dos nombres según la versión.
DATASETS_RADIANCIA = (
    "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_BRDF-Corrected_NTL",
    "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_BRDF-Corrected_NTL",
)

# Bandera de calidad. En la colección 2.0, 0 es alta calidad y de 1 a 5 son
# distintas formas de mala: 3 eclipse lunar, 4 aurora, 5 destello. 255 es
# ausencia de recuperación.
BANDERA_CALIDAD = "Mandatory_Quality_Flag"
CALIDAD_ACEPTABLE = 0

# Capa rellenada por la NASA. Donde el algoritmo principal recupera, vale
# exactamente lo mismo que él; donde no, arrastra la última recuperación buena.
# Medido sobre Cuauhtémoc el 1 y el 2 de septiembre de 2026: los 198 píxeles
# idénticos entre ambos días, diferencia máxima 0.0, con la antigüedad pasando
# de 5 a 6 días. O sea que da un valor todos los días, pero dos días seguidos
# sin recuperación no son dos observaciones: son la misma repetida.
#
# Por eso se lee junto con la antigüedad y nunca sin ella. Quien quiera serie
# continua la tiene; quien modele con rezagos filtra por antigüedad = 0. Lo que
# no se puede es publicar la capa sola, que es como un tramo plano se vuelve
# "la luz no cambió".
DATASET_RELLENADO = "Gap_Filled_DNB_BRDF-Corrected_NTL"
DATASET_ANTIGUEDAD = "Latest_High_Quality_Retrieval"

IMAGE_PATH = DATASETS_RADIANCIA[0]

# Where the downloaded HDF5 files (10–30 MB each; ~16 MB for the Mexico tiles) are staged.
#
# This used to be the literal relative path "../temp", which resolved against
# whatever the current working directory happened to be — so the same code
# wrote to a different place depending on how the process was started, and in a
# container could fill the root filesystem instead of the intended volume.
# It is now an absolute path: set NTL_TEMP_DIR to control it, otherwise it
# falls back to a subdirectory of the system temp dir, which is always writable
# and never depends on the cwd.
_DEFAULT_TEMP_DIR = Path(tempfile.gettempdir()) / "ntl"
TEMP_DIR = Path(
    _primera_definida("NTL_TEMP_DIR", "VNP46A1_TEMP_DIR") or _DEFAULT_TEMP_DIR
).expanduser().resolve()


def temp_path(filename: str) -> Path:
    """Absolute path for a staged file, creating TEMP_DIR on first use."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR / filename


# Cuántas fechas se procesan a la vez.
#
# Cada fecha en vuelo retiene en disco sus gránulos —uno por cuadrante— desde la
# primera descarga hasta que termina su último municipio, así que el pico de
# disco es:
#
#     pico ≈ MAX_FECHAS_CONCURRENTES × cuadrantes de la región × ~16 MB (tiles de México; el mayor del día ronda 30 MB)
#
# Sin este límite se lanzaba una tarea por fecha y todas descargaban a la vez:
# un año de una región de dos cuadrantes son 730 gránulos simultáneos. El
# parámetro `chunks` lo acotaba de rebote, pero existe para guardar progreso, no
# para cuidar el disco, y desde que un municipio puede necesitar cuatro
# cuadrantes en vez de uno el margen es cuatro veces menor.
MAX_FECHAS_CONCURRENTES = int(_primera_definida("NTL_MAX_FECHAS_CONCURRENTES") or 4)


# Límites hacia LAADS DAAC.
#
# NASA limita las peticiones por usuario y responde 429/503 cuando se las pasa.
# Antes el único freno era el semáforo de fechas, que cuenta fechas y no
# peticiones, y cada instancia tenía el suyo: dos corridas a la vez sumaban sus
# descargas sin que nadie las contara. Estos valores gobiernan un limitador
# global por proceso (ver core.limitador).
#
# MAX_DESCARGAS: gránulos (~10–30 MB) bajándose a la vez.
# MAX_PETICIONES_POR_SEG: ritmo de TODA petición, listados incluidos.
# MAX_INTENTOS / BACKOFF_*: reintentos ante errores transitorios, con espera
#   exponencial (base · 2^n) y tope; un Retry-After del servidor manda sobre ella.
MAX_DESCARGAS = max(1, int(_primera_definida("NTL_MAX_DESCARGAS") or 4))
# NTL_MAX_PETICIONES_POR_SEG=0 desactiva el límite de ritmo (queda el de descargas
# simultáneas y la pausa ante 429/503); un valor negativo es un error de
# configuración y se rechaza al importar en vez de fallar a mitad de una corrida.
MAX_PETICIONES_POR_SEG = float(_primera_definida("NTL_MAX_PETICIONES_POR_SEG") or 2)
if MAX_PETICIONES_POR_SEG < 0:
    raise ValueError(
        f"NTL_MAX_PETICIONES_POR_SEG={MAX_PETICIONES_POR_SEG} no es válido: usa un "
        f"número positivo, o 0 para no limitar el ritmo")
MAX_INTENTOS = max(1, int(_primera_definida("NTL_MAX_INTENTOS") or 5))
BACKOFF_BASE = float(_primera_definida("NTL_BACKOFF_BASE") or 2)
BACKOFF_TOPE = float(_primera_definida("NTL_BACKOFF_TOPE") or 120)
# Cuánto se recuerda que el listado de un día respondió 404 (s). Un día aún no
# publicado da 404 y se publica después: recordarlo para siempre lo dejaría como
# «sin datos» en cualquier proceso largo. 0 = no recordarlo.
TTL_LISTADO_404 = max(0.0, float(_primera_definida("NTL_TTL_LISTADO_404") or 600))


# Where the Parquet results are written.
#
# Esto era "../data", que además de depender del cwd escribía en el directorio
# padre: lanzar el proceso desde una subcarpeta dejaba los resultados fuera del
# proyecto. A diferencia de TEMP_DIR, el valor por omisión no es el temporal del
# sistema: son resultados, no archivos de paso, y perderlos ahí sería peor.
# Usa NTL_DATA_DIR para fijarlo.
DATA_DIR = Path(
    _primera_definida("NTL_DATA_DIR", "VNP46A1_DATA_DIR") or Path.cwd() / "data"
).expanduser().resolve()


def data_path(filename: str) -> Path:
    """Absolute path for a result file, creating DATA_DIR on first use."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / filename


def find_image_path(hdf_file) -> str:
    """
    Encuentra la ruta al dataset de radianza. Colección 5200 puede usar
    VIIRS_Grid_DNB_2d o VNP_Grid_DNB; busca alternativas si el path estándar falla.
    """
    candidates = list(DATASETS_RADIANCIA)
    for path in candidates:
        try:
            if path in hdf_file:
                return path
        except Exception:
            pass

    try:
        if "HDFEOS" in hdf_file and "GRIDS" in hdf_file["HDFEOS"]:
            grids = hdf_file["HDFEOS"]["GRIDS"]
            for grid_name in list(grids.keys()):
                grid = grids[grid_name]
                for key in list(grid.keys()):
                    if "Data" in key and "Field" in key:
                        data_fields = grid[key]
                        for field_name in list(data_fields.keys()):
                            if "Radiance" in field_name and "DNB" in field_name:
                                path = f"HDFEOS/GRIDS/{grid_name}/{key}/{field_name}"
                                try:
                                    if path in hdf_file:
                                        return path
                                except Exception:
                                    pass
    except Exception:
        pass

    return IMAGE_PATH
_DATA_ROOT = resources.files("ntl_data")

# Polígonos municipales: entrada de ntl.geometria, que los convierte en
# coberturas por píxel.
RUTA_MUNICIPIOS = str(_DATA_ROOT.joinpath("limite-de-las-alcaldias.json"))

# Coberturas ya calculadas: entrada de ntl.radianza, que las aplica a cada
# imagen diaria sin recalcular geometría.
PIXELES_MUNICIPIOS = str(_DATA_ROOT.joinpath("municipios_coordenadas_pixeles.json"))

TOKEN = os.getenv("NASA_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

# Tamaño de bloque para la descarga bloqueante
CHUNK_SIZE = 8192