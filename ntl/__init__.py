"""
Procesamiento de imágenes satelitales Black Marble (VJ146A2 y VNP46A2).

El producto por omisión es VJ146A2 (NOAA-20), el único vigente: VNP46A2 (Suomi
NPP) termina el 2026-11-02. Se cambia con NTL_PRODUCTO o con el argumento
`producto`; no son intercambiables sin calibrar (ver `core.productos`).

El paquete está dividido por responsabilidad, no por modelo de concurrencia:

    core/       configuración, modelos, utilidades y descargas
    geometria/  polígono municipal → cobertura por píxel  (una vez por municipio)
    radianza/   cobertura → métricas de luminosidad       (una vez por día)

`geometria` produce lo que `radianza` consume. La distinción entre síncrono y
asíncrono vive únicamente en `core.downloader`, que es donde importa.
"""

from .core.errores import CredencialesInvalidas
from .core.models import CoordenadasPixeles, MedicionResultado

__version__ = "0.11.0"

__all__ = ["CoordenadasPixeles", "CredencialesInvalidas", "MedicionResultado"]
