"""
El estado por event loop (limitador y cachés del downloader) no debe retener
loops: cada `asyncio.run` crea uno y, si algo lo mantiene vivo, se acumulan
selectores y descriptores en un proceso largo.
"""
import asyncio
import gc
import weakref

from aiohttp import web
import aiohttp
from aiohttp.test_utils import TestServer

import ntl.core.config as config
import ntl.core.downloader as downloader
from ntl.core import por_loop
from ntl.core.limitador import obtener_limitador

LISTADO = {"content": [{"name": "VJ146A2.A2024001.h08v07.002.x.h5",
                        "downloadsLink": "http://x/a.h5", "size": 5, "mtime": 1,
                        "resourceType": "File"}]}


def _loops_vivos(refs):
    gc.collect()
    return [r for r in refs if r() is not None]


def test_asyncio_run_repetido_no_deja_loops_vivos(monkeypatch):
    refs = []

    async def uso(dia):
        refs.append(weakref.ref(asyncio.get_running_loop()))
        # Limitador con el semáforo disputado (el Semaphore guarda el loop al
        # bloquearse) y un listado cacheado (el future referencia al loop).
        lim = obtener_limitador()

        async def descarga():
            async with lim.descarga():
                await asyncio.sleep(0.001)

        await asyncio.gather(*[descarga() for _ in range(config.MAX_DESCARGAS + 3)])
        await lim.esperar_turno()

        async def listado(request):
            return web.json_response(LISTADO)

        app = web.Application()
        app.router.add_get("/{tail:.*}", listado)
        servidor = TestServer(app)
        await servidor.start_server()
        try:
            monkeypatch.setattr(config, "LAADS_API_BASE", f"http://127.0.0.1:{servidor.port}/api")
            async with aiohttp.ClientSession() as sesion:
                assert await downloader.find_file_async(
                    sesion, 2024, dia, "h08v07", "VJ146A2") is not None
        finally:
            await servidor.close()

    for dia in range(1, 13):
        asyncio.run(uso(dia))

    assert _loops_vivos(refs) == [], "los loops de asyncio.run siguen vivos"


def test_estado_sin_dict_en_el_loop_descarta_los_loops_cerrados():
    class LoopSinDict:
        __slots__ = ("cerrado",)

        def __init__(self):
            self.cerrado = False

        def is_closed(self):
            return self.cerrado

    antes = dict(por_loop._RESPALDO)
    try:
        a, b = LoopSinDict(), LoopSinDict()
        por_loop._almacen(a)["x"] = 1
        assert por_loop._almacen(a)["x"] == 1
        a.cerrado = True
        por_loop._almacen(b)
        assert id(a) not in por_loop._RESPALDO and id(b) in por_loop._RESPALDO
    finally:
        por_loop._RESPALDO.clear()
        por_loop._RESPALDO.update(antes)


async def test_terminar_la_ultima_corrida_vacia_los_listados():
    estado = downloader._estado()
    estado.listados["k"] = object()
    downloader.iniciar_corrida()
    downloader.iniciar_corrida()
    downloader.terminar_corrida()
    assert "k" in estado.listados, "queda otra corrida en curso que comparte el listado"
    downloader.terminar_corrida()
    assert estado.listados == {}
