"""
Reintentos, validación y single-flight del downloader, contra un servidor
aiohttp local (nada de red real a NASA).
"""
import asyncio
import gzip
import re
import time

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import ntl.core.config as config
import ntl.core.downloader as downloader
from ntl import CredencialesInvalidas as CredencialesExportada
from ntl.core.downloader import download_file_async, find_file_async
from ntl.core.errores import CredencialesInvalidas
from ntl.core.limitador import obtener_limitador

H5 = b"\x89HDF\r\n\x1a\n" + b"\0" * 5000
def entrada(nombre, base="http://127.0.0.1/dl", size=len(H5), mtime=1700000000):
    """Una entrada del listado JSON de LAADS."""
    return {"downloadsLink": f"{base}/{nombre}", "mtime": mtime, "name": nombre,
            "resourceType": "File", "size": size, "status": "Online"}


def listado(*nombres, base="http://127.0.0.1/dl"):
    return {"content": [entrada(n, base) for n in nombres]}


LISTADO = listado(
    "VJ146A2.A2024001.h08v07.002.2024003022532.h5",
    "VJ146A2.A2024001.h07v07.002.2024003022532.h5",
    "VJ146A2.A2024001.h08v06.002.2024003022532.h5",
)
RUTA_LISTADO = "/api/5200/{p}/{y}/{d}/"


@pytest.fixture
def api(monkeypatch):
    """Devuelve una función que apunta la API de listados a un servidor local."""
    def apuntar(srv):
        monkeypatch.setattr(config, "LAADS_API_BASE", srv.base + "/api")
    return apuntar


class Servidor:
    """
    Servidor falso con contador de visitas por plantilla de ruta.

    Las rutas se registran con el servidor ya en marcha (aiohttp congela el
    router al arrancar), así que hay una sola ruta comodín que despacha.
    """

    def __init__(self):
        self.visitas = {}
        self._rutas = []
        app = web.Application()
        app.router.add_get("/{tail:.*}", self._despachar)
        self.server = TestServer(app)

    def ruta(self, plantilla, handler):
        patron = re.compile("^" + re.sub(r"\{\w+\}", "[^/]+", plantilla) + "$")
        self._rutas.append((plantilla, patron, handler))

    async def _despachar(self, request):
        for plantilla, patron, handler in self._rutas:
            if patron.match(request.path):
                self.visitas[plantilla] = self.visitas.get(plantilla, 0) + 1
                return await handler(request)
        return web.Response(status=404)

    async def __aenter__(self):
        await self.server.start_server()
        self.base = f"http://127.0.0.1:{self.server.port}"
        return self

    async def __aexit__(self, *exc):
        await self.server.close()


@pytest.fixture
async def sesion():
    async with aiohttp.ClientSession() as s:
        yield s


def test_credenciales_invalidas_se_exporta_desde_el_paquete():
    assert CredencialesExportada is CredencialesInvalidas


async def test_429_con_retry_after_se_respeta_y_pausa_a_todos(sesion, tmp_path, esperas):
    estado = {"n": 0}

    async def h(request):
        estado["n"] += 1
        if estado["n"] == 1:
            return web.Response(status=429, headers={"Retry-After": "0.3"})
        return web.Response(body=H5)

    async with Servidor() as srv:
        srv.ruta("/g.h5", h)
        t0 = time.perf_counter()
        res = await download_file_async(sesion, f"{srv.base}/g.h5", str(tmp_path / "g.h5"))
        dt = time.perf_counter() - t0

    assert res == str(tmp_path / "g.h5")
    assert srv.visitas["/g.h5"] == 2
    # Mandó el Retry-After del servidor, no el backoff con jitter...
    assert esperas == [0.3]
    # ...y se avisó al limitador: el segundo intento esperó la pausa común.
    assert dt >= 0.25


async def test_retry_after_se_capa_al_tope(sesion, tmp_path, esperas, monkeypatch):
    monkeypatch.setattr(config, "BACKOFF_TOPE", 0.2)
    estado = {"n": 0}

    async def h(request):
        estado["n"] += 1
        if estado["n"] == 1:
            return web.Response(status=503, headers={"Retry-After": "3600"})
        return web.Response(body=H5)

    async with Servidor() as srv:
        srv.ruta("/g.h5", h)
        res = await download_file_async(sesion, f"{srv.base}/g.h5", str(tmp_path / "g.h5"))
    assert res is not None
    assert esperas == [0.2]


@pytest.mark.parametrize("status", [401, 403])
async def test_401_403_lanzan_credenciales_sin_reintentar(sesion, tmp_path, esperas, status):
    async def h(request):
        return web.Response(status=status, text="no")

    async with Servidor() as srv:
        srv.ruta("/g.h5", h)
        with pytest.raises(CredencialesInvalidas):
            await download_file_async(sesion, f"{srv.base}/g.h5", str(tmp_path / "g.h5"))
    assert srv.visitas["/g.h5"] == 1
    assert esperas == []


async def test_html_con_200_lanza_credenciales_sin_reintentar(sesion, tmp_path, esperas):
    async def h(request):
        return web.Response(text="<html><body>Login</body></html>", content_type="text/html")

    async with Servidor() as srv:
        srv.ruta("/g.h5", h)
        with pytest.raises(CredencialesInvalidas):
            await download_file_async(sesion, f"{srv.base}/g.h5", str(tmp_path / "g.h5"))
    assert srv.visitas["/g.h5"] == 1
    assert esperas == []
    assert list(tmp_path.iterdir()) == [], "no debe quedar ni el .part"


@pytest.mark.parametrize("status", [400, 404, 410])
async def test_4xx_definitivos_devuelven_none_sin_reintentar(sesion, tmp_path, esperas, status):
    async def h(request):
        return web.Response(status=status)

    async with Servidor() as srv:
        srv.ruta("/g.h5", h)
        res = await download_file_async(sesion, f"{srv.base}/g.h5", str(tmp_path / "g.h5"))
    assert res is None
    assert srv.visitas["/g.h5"] == 1
    assert esperas == []


async def test_5xx_se_reintenta_con_backoff_y_se_rinde_con_none(sesion, tmp_path, esperas):
    async def h(request):
        return web.Response(status=500)

    async with Servidor() as srv:
        srv.ruta("/g.h5", h)
        res = await download_file_async(sesion, f"{srv.base}/g.h5", str(tmp_path / "g.h5"),
                                        max_retries=4, delay=2)
    assert res is None
    assert srv.visitas["/g.h5"] == 4
    # Full jitter: entre 0 y base·2^(n-1) en cada espera.
    assert len(esperas) == 3
    for n, espera in enumerate(esperas, start=1):
        assert 0 <= espera <= 2 * 2 ** (n - 1)


async def test_error_de_conexion_se_reintenta(sesion, tmp_path, esperas):
    # Puerto cerrado: ClientConnectorError en cada intento.
    async with Servidor() as srv:
        base = srv.base
    res = await download_file_async(sesion, f"{base}/g.h5", str(tmp_path / "g.h5"),
                                    max_retries=3)
    assert res is None
    assert len(esperas) == 2


async def test_find_file_404_es_none_y_find_file_401_lanza(sesion, api, esperas):
    async def h404(request):
        return web.Response(status=404)

    async def h401(request):
        return web.Response(status=401)

    async with Servidor() as srv:
        srv.ruta("/api/5200/VJ146A2/2024/001/", h404)
        srv.ruta("/api/5200/VJ146A2/2024/002/", h401)
        api(srv)
        assert await find_file_async(sesion, 2024, "001", "h08v07") is None
        with pytest.raises(CredencialesInvalidas):
            await find_file_async(sesion, 2024, "002", "h08v07")
    assert esperas == []


async def test_404_es_definitivo_y_se_recuerda(sesion, api, esperas):
    async def h(request):
        return web.Response(status=404)

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        api(srv)
        assert await find_file_async(sesion, 2024, "001", "h08v07") is None
        assert await find_file_async(sesion, 2024, "001", "h07v07") is None
    assert srv.visitas[RUTA_LISTADO] == 1
    assert esperas == [], "un 404 no se reintenta"


async def test_lista_vacia_es_dia_sin_publicar_y_no_se_recuerda(sesion, api):
    estado = {"n": 0}

    async def h(request):
        estado["n"] += 1
        return web.json_response({"content": []} if estado["n"] == 1 else LISTADO)

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        api(srv)
        assert await find_file_async(sesion, 2024, "001", "h08v07") is None
        assert await find_file_async(sesion, 2024, "001", "h08v07") is not None
    assert srv.visitas[RUTA_LISTADO] == 2


async def test_un_listado_por_dia_aunque_lleguen_varios_cuadrantes(sesion, api):
    async def h(request):
        await asyncio.sleep(0.05)  # da tiempo a que lleguen todos mientras vuela
        return web.json_response(LISTADO)

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        api(srv)
        urls = await asyncio.gather(*[
            find_file_async(sesion, 2024, "001", c) for c in ("h08v07", "h07v07", "h08v06")])
        # Y un llamador posterior sigue usando lo ya listado.
        otra = await find_file_async(sesion, 2024, "001", "h08v07")
        # Un día distinto sí es otra petición.
        await find_file_async(sesion, 2024, "002", "h08v07")

    assert srv.visitas[RUTA_LISTADO] == 2
    assert [u.rsplit("/", 1)[1] for u in urls] == [
        "VJ146A2.A2024001.h08v07.002.2024003022532.h5",
        "VJ146A2.A2024001.h07v07.002.2024003022532.h5",
        "VJ146A2.A2024001.h08v06.002.2024003022532.h5"]
    assert otra == urls[0]


async def test_el_single_flight_distingue_producto(sesion, api):
    async def h(request):
        producto = request.path.split("/")[3]
        return web.json_response(listado(f"{producto}.A2024001.h08v07.002.x.h5"))

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        api(srv)
        a = await find_file_async(sesion, 2024, "001", "h08v07", "VJ146A2")
        b = await find_file_async(sesion, 2024, "001", "h08v07", "VNP46A2")
    assert srv.visitas[RUTA_LISTADO] == 2
    assert "VJ146A2" in a and "VNP46A2" in b


async def test_un_listado_fallido_no_se_recuerda(sesion, api, monkeypatch, esperas):
    estado = {"n": 0}

    async def h(request):
        estado["n"] += 1
        if estado["n"] <= 2:
            return web.Response(status=500)
        return web.json_response(LISTADO)

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        api(srv)
        monkeypatch.setattr(config, "MAX_INTENTOS", 2)
        assert await find_file_async(sesion, 2024, "001", "h08v07") is None
        assert await find_file_async(sesion, 2024, "001", "h08v07") is not None


# --- Elección del archivo ---------------------------------------------------------

async def test_cuadrante_exacto_y_prefijo_del_producto(sesion, api):
    """`h08v07` no debe casar dentro de otro nombre ni con otro producto."""
    async def h(request):
        return web.json_response(listado(
            "VNP46A2.A2024001.h08v07.002.1.h5",       # otro producto
            "VJ146A2.A2024001.xh08v07x.002.1.h5",     # subcadena, sin token exacto
            "VJ146A2.A2024001.h08v070.002.1.h5",      # el token no es igual
            "VJ146A2.A2024001.h18v07.002.1.h5",
            "VJ146A2.A2024001.h08v07.002.1.h5",       # el bueno, al final
        ))

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        api(srv)
        url = await find_file_async(sesion, 2024, "001", "h08v07")
        nada = await find_file_async(sesion, 2024, "001", "h8v07")
    assert url.endswith("/VJ146A2.A2024001.h08v07.002.1.h5")
    assert nada is None


async def test_si_solo_hay_otro_producto_no_se_devuelve(sesion, api):
    async def h(request):
        return web.json_response(listado("VNP46A2.A2024001.h08v07.002.1.h5"))

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        api(srv)
        assert await find_file_async(sesion, 2024, "001", "h08v07", "VJ146A2") is None


async def test_reproceso_duplicado_elige_el_mtime_mas_reciente(sesion, api):
    async def h(request):
        return web.json_response({"content": [
            entrada("VJ146A2.A2024001.h08v07.002.2024003000000.h5", mtime=100),
            entrada("VJ146A2.A2024001.h08v07.002.2024005000000.h5", mtime=300),
            entrada("VJ146A2.A2024001.h08v07.002.2024004000000.h5", mtime=200),
        ]})

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        api(srv)
        url = await find_file_async(sesion, 2024, "001", "h08v07")
    assert url.endswith("2024005000000.h5")


async def test_mtime_manda_sobre_el_orden_del_nombre(sesion, api):
    async def h(request):
        return web.json_response({"content": [
            entrada("VJ146A2.A2024001.h08v07.002.2024009999999.h5", mtime=1),
            entrada("VJ146A2.A2024001.h08v07.002.2024001000000.h5", mtime=2),
        ]})

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        api(srv)
        url = await find_file_async(sesion, 2024, "001", "h08v07")
    assert url.endswith("2024001000000.h5")


async def test_el_listado_tolera_raiz_lista_y_entradas_raras(sesion, api):
    async def h(request):
        return web.json_response([
            "basura", {"name": "sin_enlace.h5"},
            entrada("VJ146A2.A2024001.h08v07.002.1.h5"),
        ])

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        api(srv)
        assert await find_file_async(sesion, 2024, "001", "h08v07") is not None


# --- Tamaño esperado del listado ----------------------------------------------------

async def test_el_tamano_del_listado_detecta_un_truncado_sin_content_length(
        sesion, api, tmp_path, esperas):
    """El servidor corta el cuerpo y no anuncia Content-Length: solo el listado lo delata."""
    async def granulo(request):
        resp = web.StreamResponse()  # chunked: sin Content-Length
        await resp.prepare(request)
        await resp.write(H5[:3000])  # el listado dice len(H5) = 5008
        return resp

    async def h(request):
        return web.json_response(listado("VJ146A2.A2024001.h08v07.002.1.h5", base=srv.base))

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        srv.ruta("/VJ146A2.A2024001.h08v07.002.1.h5", granulo)
        api(srv)
        url = await find_file_async(sesion, 2024, "001", "h08v07")
        res = await download_file_async(sesion, url, str(tmp_path / "g.h5"), max_retries=2)
    assert res is None
    assert not (tmp_path / "g.h5").exists()
    assert srv.visitas["/VJ146A2.A2024001.h08v07.002.1.h5"] == 2, "debía reintentar"


async def test_el_tamano_del_listado_manda_sobre_content_length(sesion, api, tmp_path, esperas):
    """Content-Length coincide con lo recibido, pero el listado dice otro tamaño."""
    async def granulo(request):
        return web.Response(body=H5[:3000])

    async def h(request):
        return web.json_response(listado("VJ146A2.A2024001.h08v07.002.1.h5", base=srv.base))

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        srv.ruta("/VJ146A2.A2024001.h08v07.002.1.h5", granulo)
        api(srv)
        url = await find_file_async(sesion, 2024, "001", "h08v07")
        res = await download_file_async(sesion, url, str(tmp_path / "g.h5"), max_retries=1)
    assert res is None


async def test_el_tamano_del_listado_acepta_el_archivo_completo(sesion, api, tmp_path):
    async def granulo(request):
        return web.Response(body=H5)

    async def h(request):
        return web.json_response(listado("VJ146A2.A2024001.h08v07.002.1.h5", base=srv.base))

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        srv.ruta("/VJ146A2.A2024001.h08v07.002.1.h5", granulo)
        api(srv)
        url = await find_file_async(sesion, 2024, "001", "h08v07")
        res = await download_file_async(sesion, url, str(tmp_path / "g.h5"))
    assert res == str(tmp_path / "g.h5")


# --- Fuera de vigencia ----------------------------------------------------------------

async def test_fecha_fuera_de_vigencia_no_pide_nada(sesion, api):
    async def h(request):
        return web.json_response(LISTADO)

    async with Servidor() as srv:
        srv.ruta(RUTA_LISTADO, h)
        api(srv)
        # 2026-307 = 2026-11-03, un día después del fin de Suomi NPP.
        assert await find_file_async(sesion, 2026, "307", "h08v07", "VNP46A2") is None
        # Antes del inicio de NOAA-20 (2018-01-05).
        assert await find_file_async(sesion, 2018, "004", "h08v07", "VJ146A2") is None
    assert srv.visitas == {}


async def test_gzip_no_descarta_un_archivo_bueno(sesion, tmp_path):
    cuerpo = gzip.compress(H5 * 40)

    async def h(request):
        return web.Response(body=cuerpo, headers={
            "Content-Encoding": "gzip", "Content-Length": str(len(cuerpo))})

    async with Servidor() as srv:
        srv.ruta("/g.h5", h)
        res = await download_file_async(sesion, f"{srv.base}/g.h5", str(tmp_path / "g.h5"),
                                        max_retries=1)
    assert res is not None
    assert (tmp_path / "g.h5").read_bytes() == H5 * 40


async def test_se_pide_identity_y_chunks_de_1mib(sesion, tmp_path):
    visto = {}

    async def h(request):
        visto["enc"] = request.headers.get("Accept-Encoding")
        return web.Response(body=H5)

    async with Servidor() as srv:
        srv.ruta("/g.h5", h)
        await download_file_async(sesion, f"{srv.base}/g.h5", str(tmp_path / "g.h5"))
    assert visto["enc"] == "identity"
    assert downloader.CHUNK_ASYNC == 1024 * 1024


def test_timeout_sin_tope_total():
    t = downloader.TIMEOUT
    assert (t.total, t.connect, t.sock_connect, t.sock_read) == (None, 30, 30, 60)


async def test_dos_descargas_del_mismo_granulo_no_se_pisan(sesion, tmp_path):
    async def h(request):
        resp = web.StreamResponse(headers={"Content-Length": str(len(H5) * 20)})
        await resp.prepare(request)
        for _ in range(20):
            await resp.write(H5)
            await asyncio.sleep(0.01)
        return resp

    async with Servidor() as srv:
        srv.ruta("/g.h5", h)
        ruta = str(tmp_path / "g.h5")
        a, b = await asyncio.gather(
            download_file_async(sesion, f"{srv.base}/g.h5", ruta),
            download_file_async(sesion, f"{srv.base}/g.h5", ruta))
    assert a == b == ruta
    assert (tmp_path / "g.h5").read_bytes() == H5 * 20
    assert [p.name for p in tmp_path.iterdir()] == ["g.h5"], "quedaron .part"


class TestBearerSoloAHostsDeNasa:
    @pytest.fixture(autouse=True)
    def token(self, monkeypatch):
        monkeypatch.setattr(downloader, "HEADERS", {"Authorization": "Bearer SECRETO"})

    @pytest.mark.parametrize("url", [
        "https://ladsweb.modaps.eosdis.nasa.gov/archive/x.h5",
        "https://urs.earthdata.nasa.gov/oauth/authorize",
        "https://data.laadsdaac.earthdatacloud.nasa.gov/x.h5",
    ])
    def test_hosts_de_nasa_por_https_llevan_token(self, url):
        assert downloader._cabeceras(url)["Authorization"] == "Bearer SECRETO"

    @pytest.mark.parametrize("url", [
        "http://ladsweb.modaps.eosdis.nasa.gov/x.h5",   # sin TLS
        "https://evil.example.com/x.h5",
        "https://nasa.gov.evil.example.com/x.h5",        # sufijo engañoso
        "https://evilnasa.gov/x.h5",
    ])
    def test_el_resto_no_lleva_token(self, url):
        assert "Authorization" not in downloader._cabeceras(url)

    def test_siempre_se_pide_identity(self):
        assert downloader._cabeceras("https://evil.example.com/")["Accept-Encoding"] == "identity"


async def test_los_redirects_se_siguen_con_urljoin(sesion, tmp_path):
    async def origen(request):
        raise web.HTTPFound("/destino/g.h5")  # relativa: hay que resolverla contra el origen

    async def destino(request):
        return web.Response(body=H5)

    async with Servidor() as srv:
        srv.ruta("/g.h5", origen)
        srv.ruta("/destino/g.h5", destino)
        res = await download_file_async(sesion, f"{srv.base}/g.h5", str(tmp_path / "g.h5"))
    assert res is not None
    assert srv.visitas["/destino/g.h5"] == 1


def test_retry_after_acepta_segundos_y_fecha_http():
    assert downloader._espera_retry_after("30") == 30
    assert downloader._espera_retry_after(None) is None
    assert downloader._espera_retry_after("basura") is None
    futura = time.time() + 60
    import email.utils
    assert 55 <= downloader._espera_retry_after(email.utils.formatdate(futura, usegmt=True)) <= 61
