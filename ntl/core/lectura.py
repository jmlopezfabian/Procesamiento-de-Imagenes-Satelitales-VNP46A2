"""
Lectura de la radianza desde el HDF5, en unidades físicas y sin valores de relleno.

El dataset viene como enteros sin signo con tres atributos que hay que respetar:
`scale_factor` y `add_offset` para llegar a nW/(cm² sr), y `_FillValue` junto con
`valid_min`/`valid_max` para saber qué píxeles no traen medición.

Ignorar el relleno no es un detalle: en el histórico hay cuatro fechas en las que
el cuadrante completo venía relleno, y como 65535 se sumó como si fuera radianza,
esos registros salieron con sumas 229 veces mayores que las normales.

A eso se suma `Mandatory_Quality_Flag`, que VNP46A2 publica por píxel y que aquí
es obligatoria: un archivo que no la traiga se rechaza en vez de procesarse sin
filtrar.
"""
from typing import Tuple

import numpy as np

from .config import (
    BANDERA_CALIDAD,
    CALIDAD_ACEPTABLE,
    DATASET_ANTIGUEDAD,
    DATASET_RELLENADO,
    PRODUCTO,
    find_image_path,
)


def _buscar_hermano(hdf_file, dataset, nombre):
    """Otro dataset del mismo grupo `Data Fields`, si existe."""
    grupo = dataset.name.rsplit("/", 1)[0]
    ruta = f"{grupo}/{nombre}"
    return hdf_file[ruta] if ruta in hdf_file else None


def _atributo(attrs, nombre, defecto=None):
    valor = attrs.get(nombre, defecto)
    if isinstance(valor, np.ndarray):
        valor = valor.item() if valor.size == 1 else valor
    return valor


def _escalar(dataset) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Aplica escala, desplazamiento y relleno a un dataset del gránulo.

    Devuelve (valores en unidades físicas, máscara de válidos, metadatos). No
    mira la bandera de calidad: eso lo decide quien llama, porque la capa
    rellenada existe precisamente para los píxeles que la bandera descarta.
    """
    crudo = dataset[()]
    attrs = dataset.attrs

    relleno = _atributo(attrs, "_FillValue")
    minimo = _atributo(attrs, "valid_min")
    maximo = _atributo(attrs, "valid_max")
    escala = float(_atributo(attrs, "scale_factor", 1.0))
    desplazamiento = float(_atributo(attrs, "add_offset", 0.0))

    valido = np.ones(crudo.shape, dtype=bool)
    if relleno is not None:
        # El relleno de VNP46A2 es -999.9 en punto flotante; compararlo con
        # igualdad exacta es frágil, así que se usa una vecindad.
        if np.issubdtype(crudo.dtype, np.floating):
            valido &= np.abs(crudo - float(relleno)) > 1e-3
        else:
            valido &= crudo != relleno
    if minimo is not None:
        valido &= crudo >= minimo
    if maximo is not None:
        valido &= crudo <= maximo

    unidades = _atributo(attrs, "units", "")
    if isinstance(unidades, bytes):
        unidades = unidades.decode(errors="replace")

    return crudo, valido, {
        "escala": escala,
        "desplazamiento": desplazamiento,
        "unidades": unidades,
    }


def _componer(crudo, valido, meta) -> np.ndarray:
    """Matriz float64 en unidades físicas, con NaN donde no hay medición."""
    salida = np.full(crudo.shape, np.nan, dtype=np.float64)
    salida[valido] = (
        crudo[valido].astype(np.float64) * meta["escala"] + meta["desplazamiento"]
    )
    return salida


def leer_rellenada(hdf_file) -> Tuple[np.ndarray, dict] | None:
    """
    La capa rellenada por la NASA, o None si el gránulo no la trae.

    **No se le aplica la bandera de calidad**, a diferencia de `leer_radianza`.
    Filtrarla por la bandera la dejaría vacía exactamente en los días para los
    que existe: donde la bandera dice 255 (sin recuperación) es donde esta capa
    arrastra el último valor bueno. Su propio `_FillValue` es el criterio.

    Léela siempre junto a `leer_antiguedad`. Un valor de esta capa sin su
    antigüedad no dice si se midió esa noche o hace seis días, y esos dos
    números se grafican igual.
    """
    dataset = _buscar_hermano(hdf_file, hdf_file[find_image_path(hdf_file)], DATASET_RELLENADO)
    if dataset is None:
        return None
    crudo, valido, meta = _escalar(dataset)
    return _componer(crudo, valido, meta), {
        "unidades": meta["unidades"],
        "fraccion_valida": float(valido.mean()),
    }


def leer_antiguedad(hdf_file) -> np.ndarray | None:
    """
    Días transcurridos desde la última recuperación de alta calidad, por píxel.

    0 significa que el valor de la capa rellenada se midió esa misma noche, y
    entonces coincide con el algoritmo principal. NaN donde el gránulo no lo
    declara. Es lo único que distingue una serie continua de una serie continua
    e inventada.
    """
    dataset = _buscar_hermano(
        hdf_file, hdf_file[find_image_path(hdf_file)], DATASET_ANTIGUEDAD
    )
    if dataset is None:
        return None
    crudo, valido, meta = _escalar(dataset)
    return _componer(crudo, valido, meta)


def leer_radianza(hdf_file) -> Tuple[np.ndarray, dict]:
    """
    Devuelve la radianza en unidades físicas, con NaN donde no hay medición.

    Es la capa del algoritmo principal, filtrada por la bandera de calidad: solo
    trae píxeles medidos esa noche. Para un valor todos los días, ver
    `leer_rellenada`, que hay que leer junto con `leer_antiguedad`.

    Args:
        hdf_file: Archivo HDF5 abierto

    Returns:
        Tuple con (matriz float64 en nW/(cm² sr), metadatos de la lectura)
    """
    dataset = hdf_file[find_image_path(hdf_file)]
    crudo, valido, meta = _escalar(dataset)
    escala, desplazamiento = meta["escala"], meta["desplazamiento"]

    # VNP46A2 marca por píxel si la recuperación sirve, y sin eso una noche
    # nublada entrega un número plausible que no es luz del suelo: el 5 de enero
    # de 2025, Iztapalapa estaba 100% nublado y VNP46A1 daba 62,838.
    #
    # Se exige la bandera en vez de seguir sin ella. Procesar un archivo sin
    # filtrar produciría una serie que parece válida y no lo es, que es
    # justamente lo que este proyecto dejó de aceptar.
    calidad = _buscar_hermano(hdf_file, dataset, BANDERA_CALIDAD)
    if calidad is None:
        raise ValueError(
            f"El archivo no trae {BANDERA_CALIDAD}: no parece un producto "
            f"{PRODUCTO}. Sin la bandera de calidad no hay forma de distinguir "
            f"una noche despejada de una completamente nublada."
        )
    valido &= calidad[()] == CALIDAD_ACEPTABLE

    radianza = _componer(crudo, valido, meta)
    unidades = meta["unidades"]

    return radianza, {
        "producto": PRODUCTO,
        "escala": escala,
        "desplazamiento": desplazamiento,
        "unidades": unidades,
        "invalidos": int((~valido).sum()),
        "fraccion_valida": float(valido.mean()),
    }
