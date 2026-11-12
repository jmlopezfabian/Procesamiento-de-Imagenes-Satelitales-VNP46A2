"""Limitador global: semáforo de descargas, token bucket y pausa común."""
import asyncio
import time

import pytest

import ntl.core.config as config
from ntl.core.limitador import Limitador, obtener_limitador, reiniciar_limitador


async def test_el_semaforo_acota_las_descargas_simultaneas():
    lim = Limitador(max_descargas=2, peticiones_por_seg=10_000)
    activas = pico = 0

    async def descarga():
        nonlocal activas, pico
        async with lim.descarga():
            activas += 1
            pico = max(pico, activas)
            await asyncio.sleep(0.01)
            activas -= 1

    await asyncio.gather(*[descarga() for _ in range(8)])
    assert pico == 2


async def test_el_bucket_limita_el_ritmo_de_peticiones():
    lim = Limitador(max_descargas=4, peticiones_por_seg=20)  # ráfaga de 20
    t0 = time.perf_counter()
    for _ in range(20):
        await lim.esperar_turno()
    assert time.perf_counter() - t0 < 0.2, "la ráfaga inicial no debería esperar"
    t0 = time.perf_counter()
    for _ in range(4):
        await lim.esperar_turno()
    # 4 fichas a 20/s: ~0.2 s con el bucket vacío.
    assert time.perf_counter() - t0 >= 0.15


async def test_penalizar_pausa_a_todos():
    lim = Limitador(max_descargas=4, peticiones_por_seg=10_000)
    lim.penalizar(0.3)
    t0 = time.perf_counter()
    await asyncio.gather(*[lim.esperar_turno() for _ in range(3)])
    assert time.perf_counter() - t0 >= 0.25


async def test_las_penalizaciones_no_se_suman():
    lim = Limitador(max_descargas=4, peticiones_por_seg=10_000)
    lim.penalizar(0.3)
    lim.penalizar(0.1)  # más corta que la vigente: no la acorta ni la alarga
    t0 = time.perf_counter()
    await lim.esperar_turno()
    assert 0.25 <= time.perf_counter() - t0 < 0.5


def test_un_limitador_por_event_loop_y_con_la_config(monkeypatch):
    monkeypatch.setattr(config, "MAX_DESCARGAS", 3)

    async def tomar():
        return obtener_limitador(), obtener_limitador()

    reiniciar_limitador()
    a1, a2 = asyncio.run(tomar())
    b1, _ = asyncio.run(tomar())
    assert a1 is a2, "dentro de un loop es el mismo"
    assert a1 is not b1, "asyncio.run repetido crea otro loop y otro limitador"
    assert a1.max_descargas == 3


def test_valores_por_omision_de_la_config(monkeypatch):
    # Los valores acordados con el servicio; cambiarlos es una decisión, no un descuido.
    import importlib
    for nombre in ("NTL_MAX_DESCARGAS", "NTL_MAX_PETICIONES_POR_SEG", "NTL_MAX_INTENTOS",
                   "NTL_BACKOFF_BASE", "NTL_BACKOFF_TOPE"):
        monkeypatch.delenv(nombre, raising=False)
    cfg = importlib.reload(config)
    try:
        assert cfg.MAX_DESCARGAS == 4
        assert cfg.MAX_PETICIONES_POR_SEG == 2
        assert cfg.MAX_INTENTOS == 5
        assert cfg.BACKOFF_BASE == 2
        assert cfg.BACKOFF_TOPE == 120
        monkeypatch.setenv("NTL_MAX_DESCARGAS", "7")
        assert importlib.reload(config).MAX_DESCARGAS == 7
    finally:
        monkeypatch.undo()
        importlib.reload(config)
