"""
Pipeline de lotes: single-flight, caché negativa, aislamiento entre corridas,
errores de red, cancelación y límite global hacia NASA.
"""
import asyncio
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
import pandas as pd
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import ntl.core.config as config
import ntl.core.downloader as downloader
import ntl.radianza.lotes as lotes
from ntl.core.errores import CredencialesInvalidas
from ntl.core.limitador import reiniciar_limitador
from ntl.radianza.lotes import SatelliteImagesAsync, cleanup_temp_files

H5 = b"\x89HDF\r\n\x1a\n" + b"\0" * 2000


class _Cobertura:
    def __init__(self, cuadrantes):
        self._c = list(cuadrantes)

    @property
    def cuadrantes(self):
        return list(self._c)

    @property
    def piezas(self):
        return {c: [(1, 1, 1.0)] for c in self._c}


def _sat(coberturas):
    with patch.object(lotes, "load_coord_data", side_effect=lambda m, _: coberturas[m]):
        return SatelliteImagesAsync(list(coberturas))


def _medicion(rutas, piezas, fecha, municipio, delete_files=True):
    return SimpleNamespace(model_dump=lambda: {"Fecha": fecha, "Municipio": municipio})


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """TEMP_DIR y DATA_DIR en tmp_path, con registro de lo que se busca y descarga."""
    temp = tmp_path / "temp"
    datos = tmp_path / "datos"
    temp.mkdir()
    datos.mkdir()
    monkeypatch.setattr(lotes, "TEMP_DIR", temp)
    monkeypatch.setattr(lotes, "temp_path", lambda n: temp / n)
    monkeypatch.setattr(lotes, "data_path", lambda n: datos / n)
    monkeypatch.setattr(lotes, "process_image_mosaico", _medicion)

    e = SimpleNamespace(temp=temp, datos=datos, busquedas=[], descargas=[],
                        fallan=set(), listado_falla=set(), retardo=0.0)

    async def find(session, year, day, cuadrante, producto=None):
        e.busquedas.append((day, cuadrante))
        if day in e.listado_falla:
            raise aiohttp.ClientConnectionError("listado caído")
        return f"http://x/{day}/{cuadrante}.h5"

    async def descarga(session, url, path):
        day, cuadrante = url.split("/")[-2], url.split("/")[-1][:-3]
        e.descargas.append((day, cuadrante))
        await asyncio.sleep(e.retardo)
        if cuadrante in e.fallan:
            return None
        Path(path).write_bytes(H5)
        return path

    monkeypatch.setattr(lotes, "find_file_async", find)
    monkeypatch.setattr(lotes, "download_file_async", descarga)
    return e


async def test_tile_fallido_se_intenta_una_sola_vez_para_n_municipios(entorno):
    entorno.fallan = {"h08v07"}
    sat = _sat({f"m{i}": _Cobertura(["h08v07"]) for i in range(6)}
               | {"doble": _Cobertura(["h08v07", "h09v07"])})
    df = await sat.run(["05-01-24"], save_progress_enabled=False)

    assert Counter(entorno.descargas) == {("005", "h08v07"): 1, ("005", "h09v07"): 1}
    assert Counter(entorno.busquedas)[("005", "h08v07")] == 1
    # Los 6 que solo necesitaban el tile caído pasan igual (sin imagen), no son fallos.
    assert sat.fallos == []
    assert len(df) == 7


async def test_fecha_repetida_descarga_una_sola_vez(entorno):
    sat = _sat({"a": _Cobertura(["h08v07"]), "b": _Cobertura(["h08v07"])})
    df = await sat.run(["05-01-24", "05-01-24", "05-01-24"], save_progress_enabled=False)
    assert entorno.descargas == [("005", "h08v07")]
    assert len(df) == 2  # dos municipios, una fecha


async def test_progreso_cuenta_fechas_unicas(entorno):
    sat = _sat({"a": _Cobertura(["h08v07"])})
    avisos = []
    await sat.run(["05-01-24", "05-01-24", "06-01-24"], save_progress_enabled=False,
                  on_progress=avisos.append)
    assert avisos == ["1/2 fechas", "2/2 fechas"]


async def test_error_de_red_en_una_fecha_no_tumba_el_lote(entorno):
    entorno.listado_falla = {"003"}
    sat = _sat({"a": _Cobertura(["h08v07"]), "b": _Cobertura(["h08v07", "h09v07"])})
    fechas = [f"{d:02d}-01-24" for d in range(1, 7)]
    df = await sat.run(fechas, save_progress_enabled=False, max_concurrentes=3)

    assert sorted(df["Fecha"].astype(str).unique()) == [
        f"2024-01-{d:02d}" for d in (1, 2, 4, 5, 6)]
    assert {f["fecha"] for f in sat.fallos} == {"2024-01-03"}
    assert {f["tipo"] for f in sat.fallos} == {"ClientConnectionError"}
    assert {f["municipio"] for f in sat.fallos} == {"a", "b"}
    # Nada quedó huérfano.
    pendientes = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert pendientes == []


async def test_credenciales_invalidas_abortan_el_lote(entorno, monkeypatch):
    async def descarga(session, url, path):
        raise CredencialesInvalidas("token malo")

    monkeypatch.setattr(lotes, "download_file_async", descarga)
    sat = _sat({"a": _Cobertura(["h08v07"])})
    with pytest.raises(CredencialesInvalidas):
        await sat.run([f"{d:02d}-01-24" for d in range(1, 9)], save_progress_enabled=False)
    assert sat.fallos == [], "no es un fallo por fecha: es la corrida entera"
    assert list(entorno.temp.iterdir()) == []
    assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []


async def test_el_loop_no_se_bloquea_mientras_se_procesa(entorno, monkeypatch):
    def lento(*a, **k):
        time.sleep(0.25)  # h5py/numpy simulados
        return _medicion(*a, **k)

    monkeypatch.setattr(lotes, "process_image_mosaico", lento)
    atrasos = []
    parar = asyncio.Event()

    async def latido():
        while not parar.is_set():
            t = time.perf_counter()
            await asyncio.sleep(0.01)
            atrasos.append(time.perf_counter() - t - 0.01)

    h = asyncio.create_task(latido())
    sat = _sat({f"m{i}": _Cobertura(["h08v07"]) for i in range(3)})
    await sat.run([f"{d:02d}-01-24" for d in range(1, 5)], save_progress_enabled=False,
                  max_concurrentes=4)
    parar.set()
    await h
    # En el loop, cada llamada congelaría ~250 ms.
    assert max(atrasos) < 0.12, f"el loop se bloqueó {max(atrasos) * 1000:.0f} ms"


async def test_cleanup_de_una_corrida_no_borra_archivos_de_otra(entorno):
    ajeno = entorno.temp / "2024-01-01_h08v07.h5"
    ajeno.write_bytes(H5)
    ajeno_part = entorno.temp / "otra.h5.123.abc.part"
    ajeno_part.write_bytes(b"x")
    otra_corrida = entorno.temp / "run_deadbeef"
    otra_corrida.mkdir()
    (otra_corrida / "x.h5").write_bytes(H5)

    sat = _sat({"a": _Cobertura(["h08v07"])})
    await sat.run(["05-01-24"], save_progress_enabled=False)

    assert ajeno.exists() and ajeno_part.exists() and (otra_corrida / "x.h5").exists()
    assert sorted(p.name for p in entorno.temp.iterdir()) == sorted(
        ["2024-01-01_h08v07.h5", "otra.h5.123.abc.part", "run_deadbeef"])


async def test_la_corrida_borra_su_directorio_aun_si_falla(entorno):
    sat = _sat({"a": _Cobertura(["h08v07"])})

    async def revienta(session, fecha, estricto=False):
        assert [p.name for p in entorno.temp.iterdir()][0].startswith("run_")
        raise RuntimeError("boom")

    with patch.object(sat, "get_measures_for_date", side_effect=revienta):
        with pytest.raises(RuntimeError):
            await sat.run(["05-01-24"], save_progress_enabled=False)
    assert list(entorno.temp.iterdir()) == []


def test_cleanup_temp_files_sigue_siendo_publica_y_borra_part(tmp_path):
    (tmp_path / "a.h5").write_bytes(b"x")
    (tmp_path / "b.h5.1.2.part").write_bytes(b"x")
    (tmp_path / "c.parquet").write_bytes(b"x")
    cleanup_temp_files(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["c.parquet"]


async def test_cancelar_run_cancela_a_las_hijas(entorno):
    entorno.retardo = 5
    sat = _sat({"a": _Cobertura(["h08v07"])})
    tarea = asyncio.create_task(
        sat.run([f"{d:02d}-01-24" for d in range(1, 9)], save_progress_enabled=False))
    await asyncio.sleep(0.2)
    assert len(entorno.descargas) > 0
    tarea.cancel()
    with pytest.raises(asyncio.CancelledError):
        await tarea
    assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []
    assert list(entorno.temp.iterdir()) == []


async def test_pool_sin_barreras_por_chunks(entorno):
    """Con chunks, una fecha lenta no frena a las siguientes: siempre hay max_concurrentes en vuelo."""
    inicio = {}

    async def descarga(session, url, path):
        day = url.split("/")[-2]
        inicio[day] = time.perf_counter()
        await asyncio.sleep(0.6 if day == "001" else 0.05)
        Path(path).write_bytes(H5)
        return path

    with patch.object(lotes, "download_file_async", descarga):
        sat = _sat({"a": _Cobertura(["h08v07"])})
        await sat.run([f"{d:02d}-01-24" for d in range(1, 9)], chunks=2,
                      save_progress_enabled=False, max_concurrentes=2)
    # Con barrera de chunk (2 fechas) la fecha 7 esperaría a la lenta; con pool
    # arranca mucho antes de que ésta termine.
    assert inicio["007"] - inicio["001"] < 0.5


async def test_checkpoint_unico_y_atomico_cada_k_fechas(entorno):
    sat = _sat({"a": _Cobertura(["h08v07"]), "b": _Cobertura(["h08v07"])})
    visto = []
    original = lotes.save_progress

    def espia(df, nombre, chunk_number=None):
        visto.append(len(df))
        return original(df, nombre, chunk_number)

    with patch.object(lotes, "save_progress", espia):
        df = await sat.run([f"{d:02d}-01-24" for d in range(1, 7)], chunks=2,
                           max_concurrentes=1)

    assert [p.name for p in entorno.datos.iterdir()] == ["multi_municipio_progress.parquet"]
    assert visto == [4, 8]  # 2 y 4 fechas completadas (la última no re-guarda lo que se devuelve)
    assert len(df) == 12


async def test_error_guarda_un_solo_archivo_de_progreso(entorno):
    sat = _sat({"a": _Cobertura(["h08v07"])})
    llamadas = {"n": 0}
    original = sat.get_measures_for_date

    async def medir(session, fecha, estricto=False):
        llamadas["n"] += 1
        if llamadas["n"] == 3:
            raise RuntimeError("bug")
        return await original(session, fecha, estricto)

    with patch.object(sat, "get_measures_for_date", side_effect=medir):
        with pytest.raises(RuntimeError):
            await sat.run([f"{d:02d}-01-24" for d in range(1, 6)], chunks=10,
                          max_concurrentes=1)
    assert [p.name for p in entorno.datos.iterdir()] == ["multi_municipio_progress.parquet"]
    assert len(pd.read_parquet(entorno.datos / "multi_municipio_progress.parquet")) == 2


async def test_resultados_en_el_orden_de_entrada(entorno):
    # La primera es la más lenta; el resultado sigue el orden pedido, no el de término.
    async def descarga(session, url, path):
        day = url.split("/")[-2]
        await asyncio.sleep({"001": 0.2, "002": 0.1}.get(day, 0))
        Path(path).write_bytes(H5)
        return path

    with patch.object(lotes, "download_file_async", descarga):
        sat = _sat({"a": _Cobertura(["h08v07"])})
        df = await sat.run(["01-01-24", "02-01-24", "03-01-24"], save_progress_enabled=False,
                           max_concurrentes=3)
    assert [str(f) for f in df["Fecha"]] == ["2024-01-01", "2024-01-02", "2024-01-03"]


async def test_el_contador_borra_cada_granulo_al_terminar_el_ultimo_municipio(entorno):
    existentes = []

    def proc(rutas, piezas, fecha, municipio, delete_files=True):
        existentes.append(all(Path(r).exists() for r in rutas.values() if r))
        return None

    with patch.object(lotes, "process_image_mosaico", proc):
        sat = _sat({"a": _Cobertura(["h08v07"]), "b": _Cobertura(["h08v07", "h09v07"])})
        await sat.run(["05-01-24"], save_progress_enabled=False)
    assert existentes == [True, True]
    assert list(entorno.temp.iterdir()) == []


# --- Límite global contra un servidor aiohttp local -------------------------------

async def test_concurrencia_hacia_nasa_acotada_con_varias_instancias(entorno, monkeypatch):
    """
    Tres instancias corriendo a la vez, cada una con sus fechas concurrentes:
    el servidor nunca ve más de NTL_MAX_DESCARGAS descargas simultáneas.
    """
    en_vuelo = {"n": 0, "pico": 0}
    listados = Counter()

    async def listado(request):
        listados[request.match_info["d"]] += 1
        base = f"http://127.0.0.1:{servidor.port}/l/{request.match_info['y']}/{request.match_info['d']}"
        return web.json_response({"content": [
            {"name": f"VJ146A2.A2024001.{c}.002.1.h5", "downloadsLink": f"{base}/{c}.h5",
             "size": len(H5) * 5, "mtime": 1, "resourceType": "File"}
            for c in ("h08v07", "h09v07")]})

    async def granulo(request):
        en_vuelo["n"] += 1
        en_vuelo["pico"] = max(en_vuelo["pico"], en_vuelo["n"])
        try:
            resp = web.StreamResponse(headers={"Content-Length": str(len(H5) * 5)})
            await resp.prepare(request)
            for _ in range(5):
                # Dormir ANTES de cada escritura: el cliente da por terminada la
                # descarga al llegar el último byte, y si el handler siguiera
                # dormido después, el servidor contaría de más.
                await asyncio.sleep(0.02)
                await resp.write(H5)
            return resp
        finally:
            en_vuelo["n"] -= 1

    app = web.Application()
    app.router.add_get("/api/5200/VJ146A2/{y}/{d}/", listado)
    app.router.add_get("/l/{y}/{d}/{f}", granulo)
    servidor = TestServer(app)
    await servidor.start_server()
    try:
        monkeypatch.setattr(config, "LAADS_API_BASE", f"http://127.0.0.1:{servidor.port}/api")
        monkeypatch.setattr(config, "MAX_DESCARGAS", 2)
        reiniciar_limitador()
        # Se restauran las funciones reales: la fixture `entorno` las había sustituido.
        monkeypatch.setattr(lotes, "find_file_async", downloader.find_file_async)
        monkeypatch.setattr(lotes, "download_file_async", downloader.download_file_async)

        sats = [_sat({f"m{i}": _Cobertura(["h08v07", "h09v07"])}) for i in range(3)]
        fechas = [f"{d:02d}-01-24" for d in range(1, 7)]
        dfs = await asyncio.gather(*[s.run(fechas, save_progress_enabled=False,
                                           max_concurrentes=3) for s in sats])
    finally:
        await servidor.close()

    assert 1 <= en_vuelo["pico"] <= 2, f"pico de descargas en el servidor: {en_vuelo['pico']}"
    assert all(len(df) == 6 for df in dfs)
    # Un listado por día para las tres instancias, no por cuadrante ni por instancia.
    assert all(n == 1 for n in listados.values()) and len(listados) == 6


# --- Producto ----------------------------------------------------------------------

async def test_el_producto_se_propaga_a_busqueda_y_a_los_nombres_temporales(entorno, monkeypatch):
    vistos = []

    async def find(session, year, day, cuadrante, producto=None):
        vistos.append(producto.nombre)
        return f"http://x/{day}/{cuadrante}.h5"

    rutas = []

    async def descarga(session, url, path):
        rutas.append(Path(path).name)
        Path(path).write_bytes(H5)
        return path

    monkeypatch.setattr(lotes, "find_file_async", find)
    monkeypatch.setattr(lotes, "download_file_async", descarga)
    with patch.object(lotes, "load_coord_data", side_effect=lambda m, _: _Cobertura(["h08v07"])):
        sat = SatelliteImagesAsync(["a"], producto="VNP46A2")
    await sat.run(["05-01-24"], save_progress_enabled=False)
    assert vistos == ["VNP46A2"]
    assert rutas == ["2024-01-05_VNP46A2_h08v07.h5"]


def test_producto_por_omision_y_validacion_al_construir(entorno, monkeypatch):
    monkeypatch.delenv("NTL_PRODUCTO", raising=False)
    assert _sat({"a": _Cobertura(["h08v07"])}).producto.nombre == "VJ146A2"
    with patch.object(lotes, "load_coord_data", side_effect=lambda m, _: _Cobertura(["h08v07"])):
        with pytest.raises(ValueError, match="VNP46A2, VJ146A2"):
            SatelliteImagesAsync(["a"], producto="VNP46A1")


async def test_fecha_fuera_de_vigencia_no_toca_la_red_y_queda_en_fallos(entorno):
    with patch.object(lotes, "load_coord_data", side_effect=lambda m, _: _Cobertura(["h08v07"])):
        sat = SatelliteImagesAsync(["a", "b"], producto="VNP46A2")
    df = await sat.run(["03-11-26", "02-11-26"], save_progress_enabled=False)
    # 2026-11-03 queda fuera; 2026-11-02 es el último día de VNP46A2.
    assert entorno.busquedas == [("306", "h08v07")]
    assert [(f["fecha"], f["municipio"], f["tipo"]) for f in sat.fallos] == [
        ("2026-11-03", "a", "ProductoFueraDeRango"), ("2026-11-03", "b", "ProductoFueraDeRango")]
    assert "VJ146A2" in sat.fallos[0]["mensaje"] and "2026-11-02" in sat.fallos[0]["mensaje"]
    assert len(df) == 2


async def test_fuera_de_vigencia_en_modo_estricto_lanza(entorno):
    with patch.object(lotes, "load_coord_data", side_effect=lambda m, _: _Cobertura(["h08v07"])):
        sat = SatelliteImagesAsync(["a"], producto="VNP46A2")
    with pytest.raises(ValueError, match="no tiene datos después de 2026-11-02"):
        await sat.run(["03-11-26"], save_progress_enabled=False, estricto=True)
    assert entorno.busquedas == []
