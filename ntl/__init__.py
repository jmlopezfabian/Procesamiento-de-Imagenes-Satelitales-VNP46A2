"""
Procesamiento de imágenes satelitales VNP46A2 (Black Marble).

El paquete está dividido por responsabilidad, no por modelo de concurrencia:

    core/       configuración, modelos, utilidades y descargas
    geometria/  polígono municipal → cobertura por píxel  (una vez por municipio)
    radianza/   cobertura → métricas de luminosidad       (una vez por día)

`geometria` produce lo que `radianza` consume. La distinción entre síncrono y
asíncrono vive únicamente en `core.downloader`, que es donde importa.
"""

from .core.models import CoordenadasPixeles, MedicionResultado

__version__ = "0.9.0"

__all__ = ["CoordenadasPixeles", "MedicionResultado"]
