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
varias veces en un mismo proceso (cada llamada es un loop nuevo).
"""
import asyncio
import time
import weakref

from . import config


class Limitador:
    def __init__(self, max_descargas: int, peticiones_por_seg: float, reloj=time.monotonic):
        self._descargas = asyncio.Semaphore(max_descargas)
        self.max_descargas = max_descargas
        self._tasa = float(peticiones_por_seg)
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


_limitadores: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def obtener_limitador() -> Limitador:
    """El limitador del event loop en curso; lo crea la primera vez."""
    loop = asyncio.get_running_loop()
    limitador = _limitadores.get(loop)
    if limitador is None:
        limitador = Limitador(config.MAX_DESCARGAS, config.MAX_PETICIONES_POR_SEG)
        _limitadores[loop] = limitador
    return limitador


def reiniciar_limitador():
    """Descarta los limitadores existentes (pensado para pruebas que cambian la config)."""
    _limitadores.clear()
