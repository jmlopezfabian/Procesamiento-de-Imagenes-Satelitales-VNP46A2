"""
VJ246A2 (NOAA-21) de punta a punta: listado LAADS -> descarga -> lectura -> métricas.

Con las funciones reales de búsqueda, descarga y extracción contra un servidor
aiohttp local que sirve gránulos sintéticos con la forma de Black Marble. Lo que
se comprueba es lo que cambia con el producto: la ruta del listado, el filtro
por nombre de archivo, el `ShortName` del gránulo y la vigencia.
"""
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import ntl.core.config as config
import ntl.core.downloader as downloader
import ntl.radianza.lotes as lotes
from ntl.radianza.lotes import SatelliteImagesAsync
from tests.test_multicuadrante import ESQUINA, FORMA, cobertura_por_cuadrante, crear_cuadrante, cuadrado

PIEZAS = cobertura_por_cuadrante(cuadrado(ESQUINA, 1.3), FORMA)


@pytest.fixture
def laads(tmp_path, monkeypatch):
    """Servidor LAADS local. `servir[(producto, cuadrante)]` = ShortName del archivo servido."""
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(lotes, "TEMP_DIR", temp)
    monkeypatch.setattr(lotes, "temp_path", lambda n: temp / n)
    monkeypatch.setattr(lotes, "data_path", lambda n: tmp_path / n)
    monkeypatch.setattr(lotes, "buscar_archivo_async", downloader.buscar_archivo_async)
    monkeypatch.setattr(lotes, "bajar_archivo_async", downloader.bajar_archivo_async)

    e = SimpleNamespace(listados=[], servir={}, temp=temp, puerto=None)
    granulos = tmp_path / "granulos"
    granulos.mkdir()

    def _bytes(prod, c):
        corto = e.servir.get((prod, c), prod)
        carpeta = granulos / f"{prod}_{corto}"
        carpeta.mkdir(exist_ok=True)
        with open(crear_cuadrante(carpeta, c, producto=corto), "rb") as f:
            return f.read()

    async def listado(request):
        p, y, d = (request.match_info[k] for k in ("p", "y", "d"))
        e.listados.append((p, y, d))
        base = f"http://127.0.0.1:{e.puerto}/g/{p}"
        # El día trae también los gránulos del otro sensor NOAA: buscar tiene que
        # quedarse solo con los del producto pedido, aunque compartan cuadrante.
        nombres = [(prod, c) for prod in ("VJ146A2", "VJ246A2") for c in PIEZAS]
        return web.json_response({"content": [
            {"name": f"{prod}.A{y}{d}.{c}.002.1.h5", "downloadsLink": f"{base}/{prod}/{c}.h5",
             "size": len(_bytes(prod, c)), "mtime": 1, "resourceType": "File"} for prod, c in nombres]})

    async def granulo(request):
        return web.Response(body=_bytes(request.match_info["prod"], request.match_info["c"]))

    def nuevo_servidor():
        # Uno por corrida: un TestServer cerrado no se puede volver a arrancar.
        app = web.Application()
        app.router.add_get("/api/5200/{p}/{y}/{d}/", listado)
        app.router.add_get("/g/{p}/{prod}/{c}.h5", granulo)
        return TestServer(app)

    return e, nuevo_servidor, monkeypatch


def _sat(producto):
    cobertura = SimpleNamespace(cuadrantes=list(PIEZAS), piezas=PIEZAS)
    with patch.object(lotes, "load_coord_data", side_effect=lambda m, _: cobertura):
        return SatelliteImagesAsync(["sintetico"], producto=producto)


async def _correr(laads, producto, fechas):
    e, nuevo_servidor, monkeypatch = laads
    servidor = nuevo_servidor()
    await servidor.start_server()
    e.puerto = servidor.port
    try:
        monkeypatch.setattr(config, "LAADS_API_BASE", f"http://127.0.0.1:{servidor.port}/api")
        sat = _sat(producto)
        df = await sat.run(fechas, save_progress_enabled=False)
    finally:
        await servidor.close()
    return sat, df


async def test_vj246a2_se_descarga_y_mide_de_punta_a_punta(laads):
    sat, df = await _correr(laads, "VJ246A2", ["05-03-24"])
    e = laads[0]

    assert sat.fallos == []
    assert len(df) == 1
    fila = df.iloc[0]
    assert fila["Producto"] == "VJ246A2"
    assert fila["Fecha"] == date(2024, 3, 5)
    assert fila["Cantidad_de_pixeles"] > 0 and fila["Media_de_radianza"] > 0
    assert fila["Fraccion_valida"] == 1.0
    # Un solo listado, en la ruta del producto (no la de VJ146A2).
    assert e.listados == [("VJ246A2", "2024", "065")]
    # Los gránulos se borran al terminar.
    assert list(e.temp.iterdir()) == []


async def test_vj246a2_y_vj146a2_dan_la_misma_medicion_sobre_el_mismo_granulo(laads):
    """Mismo formato: la única diferencia que tiene que quedar es la etiqueta."""
    _, vj2 = await _correr(laads, "VJ246A2", ["05-03-24"])
    _, vj1 = await _correr(laads, "VJ146A2", ["05-03-24"])
    metricas = ["Cantidad_de_pixeles", "Suma_de_radianza", "Media_de_radianza",
                "Percentil_50_de_radianza", "Fraccion_valida"]
    assert vj2[metricas].to_dict("records") == vj1[metricas].to_dict("records")
    assert (vj2.iloc[0]["Producto"], vj1.iloc[0]["Producto"]) == ("VJ246A2", "VJ146A2")


async def test_un_granulo_que_declara_otro_producto_no_se_etiqueta_como_vj246a2(laads):
    """Si LAADS sirviera un VJ146A2 bajo la ruta de VJ246A2, se rechaza, no se mezcla."""
    e = laads[0]
    e.servir = {("VJ246A2", c): "VJ146A2" for c in PIEZAS}
    sat, df = await _correr(laads, "VJ246A2", ["05-03-24"])
    assert df.empty
    assert [f["tipo"] for f in sat.fallos] == ["MedicionImposible"]


async def test_antes_de_noaa21_no_se_pide_a_nasa(laads):
    sat, df = await _correr(laads, "VJ246A2", ["09-02-23", "10-02-23"])
    e = laads[0]
    assert e.listados == [("VJ246A2", "2023", "041")]  # solo el 10 de febrero
    assert [(f["fecha"], f["tipo"]) for f in sat.fallos] == [("2023-02-09", "ProductoFueraDeRango")]
    assert "antes de 2023-02-10" in sat.fallos[0]["mensaje"]
    assert len(df) == 1
