"""
Descarga parcial por rangos HTTP (core.rangos).

La garantía que importa es una: con el HDF5 delgado, cada municipio sale
**idéntico** a como sale con el archivo entero, matrices incluidas. Lo demás
—bajar menos bytes, caer a la descarga completa cuando el servidor no acepta
rangos, el manejo de errores— es lo que hace que esa garantía valga en la red
real. Se prueba contra un servidor HTTP local que imita a LAADS.
"""
import asyncio
import os
import socket
import threading
import time
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import h5py
import numpy as np
import pytest

import ntl.core.config as config
import ntl.radianza.extraccion as extraccion
from ntl.core.downloader import bajar_franjas_async
from ntl.core.errores import CredencialesInvalidas, DescargaFallida
from ntl.core.limitador import reiniciar_limitador
from ntl.core.rangos import ATRIBUTO_PARCIAL, bajar_franjas, unir_intervalos
from ntl.geometria.mosaico import cobertura_por_cuadrante
from ntl.radianza.extraccion import filas_necesarias, process_image_mosaico
from tests.test_multicuadrante import crear_cuadrante, cuadrado

CUADRANTE = "h08v07"
FORMA = (120, 120)
CENTRO = (-95.0, 15.0)  # centro de h08v07 en la retícula sintética


# --- servidor que imita a LAADS ------------------------------------------------

class _Servidor:
    """
    Sirve archivos con Range. `contenido` es el de `/granulo.h5`; `archivos`
    agrega otros por nombre. `guion` fuerza las respuestas de las primeras
    peticiones: un entero es ese status, y "truncar" / "desplazar" sirven el
    rango con el cuerpo cortado a la mitad o empezando en el byte equivocado.
    `demora` (s) retrasa cada respuesta; `retry_after` va en las forzadas.
    """

    def __init__(self):
        self.contenido = b""
        self.archivos: dict[str, bytes] = {}
        self.demora = 0.0
        self.retry_after = None
        self.acepta_rangos = True
        self.existe = True
        self.guion: list[int] = []
        self.peticiones = 0
        self.bytes_servidos = 0
        servidor = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                servidor.peticiones += 1
                if servidor.demora:
                    time.sleep(servidor.demora)
                forzada = servidor.guion.pop(0) if servidor.guion else None
                if isinstance(forzada, int):
                    self.send_response(forzada)
                    if servidor.retry_after is not None:
                        self.send_header("Retry-After", str(servidor.retry_after))
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if not servidor.existe:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                nombre = self.path.lstrip("/")
                datos = servidor.archivos.get(nombre, servidor.contenido)
                rango = self.headers.get("Range")
                if rango and servidor.acepta_rangos:
                    a, b = rango.split("=")[1].split("-")
                    a, b = int(a), min(int(b), len(datos) - 1)
                    if forzada == "desplazar":
                        a, b = a + 1, min(b + 1, len(datos) - 1)
                    cuerpo = datos[a:b + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{b}/{len(datos)}")
                else:
                    cuerpo = datos
                    self.send_response(200)
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                if forzada == "truncar":
                    # Se anuncia el largo entero y se corta a la mitad.
                    self.wfile.write(cuerpo[:len(cuerpo) // 2])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                self.wfile.write(cuerpo)
                servidor.bytes_servidos += len(cuerpo)

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self._http.server_port}"
        self.url = f"{self.base}/granulo.h5"
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

    def cerrar(self):
        self._http.shutdown()
        self._http.server_close()


@pytest.fixture
def servidor():
    s = _Servidor()
    yield s
    s.cerrar()


@pytest.fixture(autouse=True)
def rapido(monkeypatch):
    monkeypatch.setattr(config, "MAX_INTENTOS", 3)
    monkeypatch.setattr(config, "BACKOFF_BASE", 0.01)
    monkeypatch.setattr(config, "BACKOFF_TOPE", 0.02)
    extraccion._LECTURAS.clear()
    reiniciar_limitador()
    yield
    extraccion._LECTURAS.clear()


def _troceado(tmp_path, nombre="granulo.h5"):
    """
    Cuadrante sintético con el troceado de los gránulos reales: franjas de
    filas comprimidas por separado. Sin troceado no habría nada que ahorrar.
    """
    plano = crear_cuadrante(tmp_path, CUADRANTE)
    ruta = tmp_path / nombre
    with h5py.File(plano, "r") as origen, h5py.File(ruta, "w") as salida:
        for k, v in origen.attrs.items():
            salida.attrs[k] = v

        def copiar(n, o):
            if isinstance(o, h5py.Group):
                g = salida.require_group(n)
                for k, v in o.attrs.items():
                    g.attrs[k] = v
            elif o.shape == FORMA:
                datos = o[()]
                if datos.dtype.kind == "f":
                    # Ruido, como la radianza real: un patrón regular se
                    # comprime a casi nada y no dejaría nada que ahorrar.
                    ruido = np.random.default_rng(0).uniform(0, 1, FORMA)
                    datos = np.where(datos > 0, datos + ruido, datos).astype(datos.dtype)
                d = salida.create_dataset(n, data=datos, chunks=(10, FORMA[1]),
                                          compression="gzip")
                for k, v in o.attrs.items():
                    d.attrs[k] = v
            else:
                d = salida.create_dataset(n, data=o[()])
        origen.visititems(copiar)
    return str(ruta)


def _municipios():
    """Dos municipios al norte y uno al sur, con franjas vacías entre ellos."""
    return {
        "norte_a": cobertura_por_cuadrante(cuadrado((-97.0, 18.5), 0.6), FORMA),
        "norte_b": cobertura_por_cuadrante(cuadrado((-93.0, 18.0), 0.8), FORMA),
        "sur": cobertura_por_cuadrante(cuadrado((-95.0, 12.0), 0.7), FORMA),
    }


def _filas_para(municipios):
    return lambda forma: filas_necesarias(list(municipios.values()), CUADRANTE, forma)


def _procesar(ruta, municipios):
    extraccion._LECTURAS.clear()
    return {n: process_image_mosaico({CUADRANTE: ruta}, p, date(2024, 1, 1), n,
                                     delete_files=False).model_dump(mode="json")
            for n, p in municipios.items()}


# --- la garantía -----------------------------------------------------------------

def test_el_delgado_da_exactamente_lo_mismo_que_el_completo(tmp_path, servidor):
    completo = _troceado(tmp_path)
    servidor.contenido = open(completo, "rb").read()
    municipios = _municipios()

    delgado = str(tmp_path / "delgado.h5")
    stats = bajar_franjas(servidor.url, delgado, _filas_para(municipios), bloque=1024)

    assert _procesar(delgado, municipios) == _procesar(completo, municipios)
    # Y para eso bajó una parte del archivo, no todo.
    assert stats["bytes"] < 0.8 * len(servidor.contenido)


def test_el_delgado_conserva_forma_y_se_declara_parcial(tmp_path, servidor):
    completo = _troceado(tmp_path)
    servidor.contenido = open(completo, "rb").read()
    delgado = str(tmp_path / "delgado.h5")
    stats = bajar_franjas(servidor.url, delgado, lambda forma: [(20, 40)], bloque=4096)

    with h5py.File(delgado, "r") as f, h5py.File(completo, "r") as g:
        assert [list(i) for i in f.attrs[ATRIBUTO_PARCIAL]] == [[20, 40]]
        capa = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_BRDF-Corrected_NTL"
        assert f[capa].shape == g[capa].shape
        np.testing.assert_array_equal(f[capa][20:40], g[capa][20:40])
        # Fuera de las filas pedidas no hay datos inventados: el relleno del archivo.
        assert (f[capa][:20] == g[capa].fillvalue).all()
        assert f["HDFEOS INFORMATION/StructMetadata.0"][()] == \
            g["HDFEOS INFORMATION/StructMetadata.0"][()]
    assert stats["filas"] == [(20, 40)]


def test_las_filas_cubren_el_bounding_box_y_no_solo_los_pixeles():
    """
    Un municipio en dos partes separadas por filas sin territorio: el recorte
    guardado abarca el rectángulo entero, así que esas filas también se bajan.
    """
    norte = cobertura_por_cuadrante(cuadrado((-95.0, 18.0), 0.5), FORMA)
    sur = cobertura_por_cuadrante(cuadrado((-95.0, 13.0), 0.5), FORMA)
    partido = {CUADRANTE: norte[CUADRANTE] + sur[CUADRANTE]}
    ys = [y for _, y, _ in partido[CUADRANTE]]

    assert filas_necesarias([partido], CUADRANTE, FORMA) == [(min(ys), max(ys) + 1)]


def test_un_municipio_de_otro_cuadrante_no_pide_filas():
    otro = {"h07v06": [(5, 5, 1.0)]}
    assert filas_necesarias([otro], CUADRANTE, FORMA) == []


def test_unir_intervalos():
    assert unir_intervalos([(5, 9), (0, 3), (2, 4), (9, 12), (20, 20)]) == [(0, 4), (5, 12)]


# --- la red real no siempre coopera ----------------------------------------------

async def _async(servidor, path, municipios):
    async with aiohttp.ClientSession() as s:
        return await bajar_franjas_async(s, servidor.url, path, _filas_para(municipios))


def test_si_el_servidor_ignora_el_rango_baja_el_archivo_entero(tmp_path, servidor):
    completo = _troceado(tmp_path)
    servidor.contenido = open(completo, "rb").read()
    servidor.acepta_rangos = False
    destino = str(tmp_path / "bajado.h5")

    assert asyncio.run(_async(servidor, destino, _municipios())) == destino
    assert open(destino, "rb").read() == servidor.contenido
    with h5py.File(destino, "r") as f:
        assert ATRIBUTO_PARCIAL not in f.attrs


def test_un_granulo_que_no_existe_es_none(tmp_path, servidor):
    servidor.existe = False
    destino = str(tmp_path / "x.h5")
    assert asyncio.run(_async(servidor, destino, _municipios())) is None
    assert not os.path.exists(destino)


def test_un_401_aislado_se_reintenta(tmp_path, servidor):
    """LAADS ha dado 401 pasajeros con un token vigente; uno solo no aborta."""
    completo = _troceado(tmp_path)
    servidor.contenido = open(completo, "rb").read()
    servidor.guion = [401]
    destino = str(tmp_path / "d.h5")
    assert bajar_franjas(servidor.url, destino, _filas_para(_municipios()), bloque=4096)
    assert os.path.exists(destino)


def test_dos_401_seguidos_son_credenciales_invalidas(tmp_path, servidor):
    servidor.contenido = open(_troceado(tmp_path), "rb").read()
    servidor.guion = [401, 401]
    destino = str(tmp_path / "d.h5")
    with pytest.raises(CredencialesInvalidas):
        bajar_franjas(servidor.url, destino, _filas_para(_municipios()), bloque=4096)
    assert not os.path.exists(destino)
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".part")]


def test_un_401_a_media_lectura_llega_como_credenciales_invalidas(tmp_path, servidor):
    """
    Dentro de h5py: el error lo lanza el archivo remoto mientras h5py lee, y
    h5py lo envuelve. Tiene que salir con su tipo, o el lote lo anotaría como
    un fallo cualquiera y seguiría golpeando a NASA con un token muerto.
    """
    servidor.contenido = open(_troceado(tmp_path), "rb").read()
    destino = str(tmp_path / "d.h5")

    # El primer bloque sale bien; a partir de ahí, solo 401.
    def filas(forma):
        servidor.guion = [401] * 50
        return [(0, 120)]

    with pytest.raises(CredencialesInvalidas):
        bajar_franjas(servidor.url, destino, filas, bloque=1024)
    assert not os.path.exists(destino)


def test_un_503_se_reintenta(tmp_path, servidor):
    completo = _troceado(tmp_path)
    servidor.contenido = open(completo, "rb").read()
    servidor.guion = [503]
    destino = str(tmp_path / "d.h5")
    assert bajar_franjas(servidor.url, destino, _filas_para(_municipios()), bloque=4096)


def test_503_persistente_agota_los_intentos(tmp_path, servidor):
    servidor.contenido = open(_troceado(tmp_path), "rb").read()
    servidor.guion = [503] * 10
    with pytest.raises(DescargaFallida):
        bajar_franjas(servidor.url, str(tmp_path / "d.h5"), _filas_para(_municipios()))


# --- integración con el lote -------------------------------------------------------

def test_el_lote_usa_la_descarga_parcial_solo_si_esta_activada(monkeypatch, tmp_path):
    import ntl.radianza.lotes as lotes

    llamadas = []

    async def parcial(session, url, path, filas_para):
        llamadas.append(("parcial", filas_para(FORMA)))
        return path

    async def completa(session, url, path):
        llamadas.append(("completa", None))
        return path

    async def buscar(*a, **k):
        return "https://ladsweb.example/x.h5"

    class Cob:
        cuadrantes = [CUADRANTE]
        piezas = {CUADRANTE: [(1, 7, 1.0), (3, 9, 0.5)]}

    monkeypatch.setattr(lotes, "load_coord_data", lambda m, _: Cob())
    monkeypatch.setattr(lotes, "buscar_archivo_async", buscar)
    monkeypatch.setattr(lotes, "bajar_franjas_async", parcial)
    monkeypatch.setattr(lotes, "bajar_archivo_async", completa)
    monkeypatch.setattr(lotes, "temp_path", lambda n: tmp_path / n)
    sat = lotes.SatelliteImagesAsync(["iztapalapa"], producto="VNP46A2")

    for activada in (False, True):
        monkeypatch.setattr(config, "DESCARGA_PARCIAL", activada)
        asyncio.run(sat._bajar_h5(None, 2024, "001", CUADRANTE, date(2024, 1, 1), "k"))

    assert llamadas == [("completa", None), ("parcial", [(7, 10)])]


def test_un_cuerpo_cortado_a_media_respuesta_se_reintenta(tmp_path, servidor):
    """
    Un corte a media respuesta no es ConnectionError para requests: es
    ChunkedEncodingError. Antes escapaba sin reintentarse y perdía la fecha.
    """
    completo = _troceado(tmp_path)
    servidor.contenido = open(completo, "rb").read()
    servidor.guion = ["truncar"]
    municipios = _municipios()
    delgado = str(tmp_path / "d.h5")
    assert bajar_franjas(servidor.url, delgado, _filas_para(municipios), bloque=1024)
    assert _procesar(delgado, municipios) == _procesar(completo, municipios)


def test_un_rango_desplazado_no_se_acepta(tmp_path, servidor):
    """Un 206 que empieza en otro byte corrompería la lectura: se reintenta."""
    completo = _troceado(tmp_path)
    servidor.contenido = open(completo, "rb").read()
    servidor.guion = ["desplazar"]
    municipios = _municipios()
    delgado = str(tmp_path / "d.h5")
    assert bajar_franjas(servidor.url, delgado, _filas_para(municipios), bloque=1024)
    assert _procesar(delgado, municipios) == _procesar(completo, municipios)


def test_un_rango_siempre_desplazado_agota_los_intentos(tmp_path, servidor):
    servidor.contenido = open(_troceado(tmp_path), "rb").read()
    servidor.guion = ["desplazar"] * 10
    with pytest.raises(DescargaFallida):
        bajar_franjas(servidor.url, str(tmp_path / "d.h5"), _filas_para(_municipios()))


def test_un_429_frena_al_limitador_de_todos(tmp_path, servidor):
    servidor.contenido = open(_troceado(tmp_path), "rb").read()
    servidor.guion = [429]
    servidor.retry_after = 7
    pausas = []
    assert bajar_franjas(servidor.url, str(tmp_path / "d.h5"), _filas_para(_municipios()),
                         bloque=4096, penalizar=pausas.append)
    # La espera que pidió el servidor (acotada por el tope de backoff).
    assert pausas == [pytest.approx(config.BACKOFF_TOPE)]


def test_cancelar_detiene_el_hilo_y_no_deja_archivo(tmp_path, servidor, monkeypatch):
    """
    `to_thread` no se puede cancelar. Antes, al abortar un lote, el hilo seguía
    pidiendo rangos sin lugar de descarga y podía escribir en un directorio de
    corrida ya borrado. Ahora se le avisa y se espera a que suelte.
    """
    servidor.contenido = open(_troceado(tmp_path), "rb").read()
    servidor.demora = 0.05
    # Bloques chicos: ~70 peticiones, así la descarga sigue en curso al cancelar.
    monkeypatch.setattr(config, "BLOQUE_RANGO", 1024)
    destino = str(tmp_path / "d.h5")

    async def correr():
        async with aiohttp.ClientSession() as s:
            tarea = asyncio.ensure_future(
                bajar_franjas_async(s, servidor.url, destino, lambda forma: [(0, 120)]))
            await asyncio.sleep(0.3)
            tarea.cancel()
            with pytest.raises(asyncio.CancelledError):
                await tarea
            antes = servidor.peticiones
            await asyncio.sleep(0.3)
            return antes, servidor.peticiones

    antes, despues = asyncio.run(correr())
    assert 0 < antes < 60  # se canceló a media descarga
    assert despues == antes  # y el hilo ya no pidió nada más
    assert not os.path.exists(destino)
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".part")]


def test_mosaico_de_varios_cuadrantes_da_lo_mismo(tmp_path, servidor):
    """
    Un municipio repartido entre cuatro cuadrantes: cada uno con sus filas en
    el marco de referencia. El recorte guardado tiene que salir igual.
    """
    from tests.test_multicuadrante import CUATRO, ESQUINA

    piezas = cobertura_por_cuadrante(cuadrado(ESQUINA, 1.3), FORMA)
    completos, delgados = {}, {}
    for c in CUATRO:
        d = tmp_path / c
        d.mkdir()
        plano = crear_cuadrante(d, c)
        servidor.archivos[f"{c}.h5"] = open(plano, "rb").read()
        completos[c] = plano
        delgados[c] = str(tmp_path / f"{c}_delgado.h5")
        bajar_franjas(f"{servidor.base}/{c}.h5", delgados[c],
                      lambda forma, c=c: filas_necesarias([piezas], c, forma), bloque=1024)
        with h5py.File(delgados[c], "r") as f:
            # De verdad es parcial: no se pidieron todas las filas.
            assert f.attrs[ATRIBUTO_PARCIAL].tolist() != [[0, FORMA[0]]]

    def procesar(rutas):
        extraccion._LECTURAS.clear()
        return process_image_mosaico(rutas, piezas, date(2024, 1, 1), "mosaico",
                                     delete_files=False).model_dump(mode="json")

    assert procesar(delgados) == procesar(completos)
