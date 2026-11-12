"""
Productos de Black Marble diarios que la librería sabe leer.

VNP46A2 (Suomi NPP) y VJ146A2 (NOAA-20 / JPSS-1) comparten algoritmo, retícula,
colección (5200) y formato HDF5: las mismas capas, dtypes, `_FillValue` y
`scale_factor`. Suomi NPP dejó de entregar datos el 2026-11-02, así que
VNP46A2 termina en el día 2026-306 y su sucesor vigente es VJ146A2.

NO son intercambiables sin calibrar. Medido una sola noche: sobre CDMX, VJ1 dio
~16-23% más que VNP en píxeles de calidad 0, y en el tile completo ~17-27%
menos. Es una noche, no una calibración: solo indica que hace falta calibrar en
el traslape 2018-2026 antes de empalmar series. Por eso cada registro lleva su
`Producto` (el que declara el archivo) y la librería nunca mezcla productos ni
elige uno por fecha a escondidas: el que pide quien llama es el que se usa, y
si la fecha cae fuera de su vigencia se dice en vez de pedirlo a NASA.

VNP46A1 sigue fuera a propósito (ver el comentario en `config`): no trae
`Mandatory_Quality_Flag`.
"""
from dataclasses import dataclass
from datetime import date

from . import config


@dataclass(frozen=True)
class Producto:
    nombre: str
    satelite: str
    plataforma: str
    inicio: date
    fin: date | None = None  # último día con datos, inclusive; None = vigente
    coleccion: int = 5200
    descripcion: str = ""

    def url_listado(self, year, day) -> str:
        """Endpoint JSON de LAADS con los archivos de un día (público; acepta Bearer)."""
        return (f"{config.LAADS_API_BASE}/{self.coleccion}/{self.nombre}/"
                f"{int(year)}/{int(day):03d}/")

    def motivo_fuera_de_rango(self, fecha: date) -> str | None:
        """None si `fecha` está en [inicio, fin]; si no, el porqué, listo para mostrar."""
        if fecha < self.inicio:
            return (f"{self.nombre} no tiene datos antes de {self.inicio.isoformat()} "
                    f"({fecha.isoformat()} es anterior)")
        if self.fin is not None and fecha > self.fin:
            sucesores = [n for n, p in PRODUCTOS.items() if p.fin is None]
            sugerencia = f"; usar {' o '.join(sucesores)}" if sucesores else ""
            return (f"{self.nombre} no tiene datos después de "
                    f"{self.fin.isoformat()}{sugerencia}")
        return None


PRODUCTOS: dict[str, Producto] = {
    "VNP46A2": Producto(
        nombre="VNP46A2", satelite="Suomi NPP", plataforma="Suomi-NPP",
        inicio=date(2012, 1, 19), fin=date(2026, 11, 2),
        descripcion="Black Marble diario corregido por BRDF lunar y atmósfera, VIIRS/Suomi NPP",
    ),
    "VJ146A2": Producto(
        nombre="VJ146A2", satelite="NOAA-20", plataforma="JPSS-1",
        inicio=date(2018, 1, 5), fin=None,
        descripcion="Black Marble diario corregido por BRDF lunar y atmósfera, VIIRS/NOAA-20",
    ),
}

# El único vigente: el de Suomi NPP ya no recibe datos nuevos.
PRODUCTO_POR_OMISION = "VJ146A2"


def producto_por_omision() -> Producto:
    """NTL_PRODUCTO si está definida (se lee al llamar, no al importar); si no, VJ146A2."""
    return obtener_producto(config._primera_definida("NTL_PRODUCTO") or PRODUCTO_POR_OMISION)


def obtener_producto(producto: "str | Producto | None" = None) -> Producto:
    """
    Resuelve un nombre, un `Producto` o None (el por omisión) a un `Producto`.

    Lanza ValueError listando los soportados si el nombre no existe.
    """
    if producto is None:
        return producto_por_omision()
    if isinstance(producto, Producto):
        return producto
    clave = str(producto).strip().upper()
    if clave not in PRODUCTOS:
        raise ValueError(
            f"Producto no soportado: {producto!r}. Soportados: "
            f"{', '.join(PRODUCTOS)}. VNP46A1 no se admite: no trae bandera de calidad.")
    return PRODUCTOS[clave]
