"""
Limitador global de peticiones a LAADS.

LAADS DAAC limita las peticiones por usuario y contesta 429/503 cuando se
excede. Tres mecanismos, compartidos por todo el proceso:

- un semáforo de descargas pesadas (gránulos de ~10–30 MB),
- un token bucket para TODA petición, listados incluidos, que son baratos pero
  cuentan igual para el límite de NASA,
- `penalizar(segundos)`, que pausa a todos cuando el servidor pidió esperar:
  si una descarga recibe un 429 y las demás siguen golpeando, el castigo se
  alarga para todas.

Se crea uno por event loop y de forma perezosa: los primitivos de asyncio se
atan al loop donde se usan por primera vez, y `asyncio.run` puede llamarse
varias veces en un mismo proceso (cada llamada es un loop nuevo). El estado
cuelga del propio loop (ver por_loop) para que no lo retenga tras cerrarse.

`peticiones_por_seg <= 0` significa sin límite de ritmo (el semáforo de descargas
y las pausas por 429/503 siguen vigentes).
"""
import asyncio
import time

from . import config
from .por_loop import estado_del_loop, olvidar_estado


class Limitador:
    def __init__(self, max_descargas: int, peticiones_por_seg: float, reloj=time.monotonic):
        self._descargas = asyncio.Semaphore(max_descargas)
        self.max_descargas = max_descargas
        # <= 0: sin bucket. Dividir entre la tasa para saber cuánto esperar
        # daba ZeroDivisionError con NTL_MAX_PETICIONES_POR_SEG=0.
        self._tasa = max(0.0, float(peticiones_por_seg))
        # Ráfaga de a lo sumo un segundo de peticiones, y nunca menos de una.
        self._capacidad = max(1.0, self._tasa)
        self._fichas = self._capacidad
        self._reloj = reloj
        self._ultimo = reloj()
        self._pausa_hasta = 0.0
        # Quien espera su ficha toma este candado: así los turnos salen en el
        # orden de llegada y no todos despiertan a la vez al rellenarse.
        self._turno = asyncio.Lock()

    def descarga(self):
        """Contexto asíncrono que acota las descargas pesadas simultáneas."""
        return self._descargas

    async def esperar_turno(self):
        """Espera hasta poder hacer una petición (ficha del bucket y sin pausa)."""
        async with self._turno:
            while True:
                ahora = self._reloj()
                if ahora < self._pausa_hasta:
                    await asyncio.sleep(self._pausa_hasta - ahora)
                    continue
                if self._tasa <= 0:
                    return
                self._fichas = min(self._capacidad,
                                   self._fichas + (ahora - self._ultimo) * self._tasa)
                self._ultimo = ahora
                if self._fichas >= 1.0:
                    self._fichas -= 1.0
                    return
                await asyncio.sleep((1.0 - self._fichas) / self._tasa)

    def penalizar(self, segundos: float):
        """Pausa todas las peticiones, de todos, durante `segundos` (no se acumula)."""
        ahora = self._reloj()
        self._pausa_hasta = max(self._pausa_hasta, ahora + max(0.0, segundos))
        # Al terminar la pausa se arranca con el bucket vacío, no con una ráfaga.
        self._fichas = 0.0
        self._ultimo = self._pausa_hasta


_CLAVE = "limitador"


def obtener_limitador() -> Limitador:
    """El limitador del event loop en curso; lo crea la primera vez."""
    return estado_del_loop(
        _CLAVE,
        lambda: Limitador(config.MAX_DESCARGAS, config.MAX_PETICIONES_POR_SEG))


def reiniciar_limitador():
    """Descarta los limitadores existentes (pensado para pruebas que cambian la config)."""
    olvidar_estado(_CLAVE)
