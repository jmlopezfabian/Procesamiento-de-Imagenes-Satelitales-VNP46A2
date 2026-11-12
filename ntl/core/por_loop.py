"""
Estado por event loop que no retiene al loop.

El limitador y las cachés del downloader guardan primitivos de asyncio
(Semaphore, Lock, futures) que se atan al loop donde nacen. Guardarlos en un
`WeakKeyDictionary[loop, ...]` parece suficiente y no lo es: el valor referencia
al loop, así que la clave «débil» nunca queda sin referentes fuertes y cada
`asyncio.run` dejaba su loop vivo para siempre (con su selector y sus
descriptores).

Aquí el estado cuelga del propio loop (un atributo): loop y estado forman un
ciclo que el recolector sabe liberar, y cuando el loop se descarta se va con él.
Si el loop no admite atributos (algunas implementaciones en extensión), se usa
un diccionario por id(loop) que descarta los loops ya cerrados en cada acceso.
"""
import asyncio
import weakref

_ATRIBUTO = "_ntl_estado_por_loop"
# Loops que tienen estado en un atributo, para poder reiniciarlo desde código
# síncrono sin retenerlos.
_CONOCIDOS: "weakref.WeakSet" = weakref.WeakSet()
# Solo para loops sin __dict__: id -> (loop, estado). Retiene al loop hasta que
# se cierre y alguien vuelva a pasar por aquí.
_RESPALDO: dict = {}


def _purgar_cerrados():
    for clave in [k for k, (loop, _) in _RESPALDO.items() if loop.is_closed()]:
        del _RESPALDO[clave]


def _almacen(loop) -> dict:
    try:
        almacen = loop.__dict__.get(_ATRIBUTO)
        if almacen is None:
            almacen = loop.__dict__[_ATRIBUTO] = {}
            _CONOCIDOS.add(loop)
        return almacen
    except AttributeError:
        _purgar_cerrados()
        return _RESPALDO.setdefault(id(loop), (loop, {}))[1]


def estado_del_loop(clave: str, fabrica):
    """El estado `clave` del loop en curso; lo crea con `fabrica()` la primera vez."""
    almacen = _almacen(asyncio.get_running_loop())
    if clave not in almacen:
        almacen[clave] = fabrica()
    return almacen[clave]


def olvidar_estado(clave: str):
    """Descarta el estado `clave` en todos los loops (para pruebas y reinicios)."""
    for loop in list(_CONOCIDOS):
        getattr(loop, "__dict__", {}).get(_ATRIBUTO, {}).pop(clave, None)
    for _, almacen in list(_RESPALDO.values()):
        almacen.pop(clave, None)
