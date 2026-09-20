"""Tests for ntl downloader with mocked HTTP."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ntl.core.downloader import find_file_async, download_file_async


def _make_resp(status=200, text="", content_bytes=None, content_length="auto"):
    """
    Respuesta falsa.

    `headers` es un dict de verdad y no un MagicMock: aiohttp devuelve cadenas, y
    con un MagicMock `int(headers.get("Content-Length"))` daba 1 sin protestar,
    que es justo el tipo de mock que hace pasar una prueba por lo que no es.
    `content_length=None` simula un servidor que no anuncia el tamaño; un número,
    uno que miente o corta la conexión.
    """
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=text)
    if content_bytes is not None:
        resp.content.read = AsyncMock(side_effect=[content_bytes, b""])
    else:
        resp.content.read = AsyncMock(return_value=b"")
    if content_length == "auto":
        content_length = len(content_bytes) if content_bytes is not None else 0
    resp.headers = {} if content_length is None else {"Content-Length": str(content_length)}
    return resp


@pytest.mark.asyncio
class TestFindFileAsync:
    async def test_returns_url_when_html_has_h5_link(self, sample_directory_html):
        session = MagicMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=_make_resp(200, sample_directory_html))
        ctx.__aexit__ = AsyncMock(return_value=None)
        session.get = MagicMock(return_value=ctx)
        url = await find_file_async(session, 2024, 1, "h08v07")
        assert url is not None
        assert "h08v07" in url
        assert url.endswith(".h5")

    async def test_returns_none_when_status_not_200(self):
        session = MagicMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=_make_resp(404, ""))
        ctx.__aexit__ = AsyncMock(return_value=None)
        session.get = MagicMock(return_value=ctx)
        url = await find_file_async(session, 2024, 1, "h08v07")
        assert url is None

    async def test_returns_none_when_no_matching_link(self):
        session = MagicMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(
            return_value=_make_resp(200, "<html><body><a href='other.h5'>x</a></body></html>")
        )
        ctx.__aexit__ = AsyncMock(return_value=None)
        session.get = MagicMock(return_value=ctx)
        url = await find_file_async(session, 2024, 1, "h99v99")
        assert url is None


@pytest.mark.asyncio
class TestDownloadFileAsync:
    async def test_returns_path_when_download_ok(self, tmp_path):
        path = str(tmp_path / "out.h5")
        session = MagicMock()
        ctx = MagicMock()
        resp = _make_resp(200, "", content_bytes=b"binary content here")
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=None)
        session.get = MagicMock(return_value=ctx)
        result = await download_file_async(session, "http://example.com/file.h5", path)
        assert result == path
        assert (tmp_path / "out.h5").read_bytes() == b"binary content here"

    async def test_returns_none_when_status_not_200(self, tmp_path):
        path = str(tmp_path / "out.h5")
        session = MagicMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=_make_resp(404, ""))
        ctx.__aexit__ = AsyncMock(return_value=None)
        session.get = MagicMock(return_value=ctx)
        result = await download_file_async(session, "http://example.com/file.h5", path)
        assert result is None


@pytest.mark.asyncio
class TestUnaDescargaAMediasNoSeDaPorBuena:
    """
    Se escribía directo sobre la ruta final. Una conexión cortada dejaba ahí un
    gránulo truncado que el cache daba por bueno y servía al resto de municipios
    de esa fecha. h5py no lo lee —se comprobó: falla con OSError incluso
    truncando solo el 1%—, así que no salían métricas malas; salía algo peor de
    diagnosticar: el municipio se registraba como si le faltara la imagen, con
    Fraccion_valida < 1, en vez de reintentar la descarga. Y como el archivo
    corrupto se quedaba en su sitio, el reintento no llegaba nunca.
    """

    def _sesion(self, resp):
        session = MagicMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=None)
        session.get = MagicMock(return_value=ctx)
        return session

    async def test_si_llegan_menos_bytes_de_los_anunciados_no_hay_archivo(self, tmp_path):
        path = str(tmp_path / "out.h5")
        # El servidor anuncia 500 MB y manda 19 bytes.
        resp = _make_resp(200, "", content_bytes=b"binary content here",
                          content_length=500_000_000)
        result = await download_file_async(
            self._sesion(resp), "http://example.com/f.h5", path, max_retries=1)

        assert result is None
        assert not (tmp_path / "out.h5").exists(), \
            "quedó un gránulo truncado con pinta de estar entero"
        assert list(tmp_path.glob("*.part")) == [], "quedó el archivo temporal"

    async def test_no_deja_rastro_si_la_conexion_se_corta(self, tmp_path):
        path = str(tmp_path / "out.h5")
        resp = _make_resp(200, "", content_length=1000)
        resp.content.read = AsyncMock(side_effect=[b"algo", ConnectionResetError()])

        result = await download_file_async(
            self._sesion(resp), "http://example.com/f.h5", path, max_retries=1, delay=0)

        assert result is None
        assert list(tmp_path.iterdir()) == [], f"quedaron restos: {list(tmp_path.iterdir())}"

    async def test_un_servidor_sin_content_length_sigue_funcionando(self, tmp_path):
        """No todos los servidores lo anuncian; sin él no se puede comprobar."""
        path = str(tmp_path / "out.h5")
        resp = _make_resp(200, "", content_bytes=b"binary content here",
                          content_length=None)
        result = await download_file_async(
            self._sesion(resp), "http://example.com/f.h5", path)

        assert result == path
        assert (tmp_path / "out.h5").read_bytes() == b"binary content here"

    async def test_en_la_ruta_final_nunca_aparece_un_archivo_a_medias(self, tmp_path):
        """
        El renombrado es atómico. Mientras se descarga, quien mire la ruta final
        no ve nada; nunca ve un archivo creciendo que podría abrir por error.
        """
        path = str(tmp_path / "out.h5")
        visto = []

        async def leer_espiando(n=8192):
            visto.append(sorted(p.name for p in tmp_path.iterdir()))
            return b"" if len(visto) > 2 else b"x" * 10

        resp = _make_resp(200, "", content_length=20)
        resp.content.read = AsyncMock(side_effect=leer_espiando)

        result = await download_file_async(
            self._sesion(resp), "http://example.com/f.h5", path)

        assert result == path
        assert all("out.h5" not in nombres for nombres in visto), \
            f"la ruta final existía a media descarga: {visto}"
        assert all(any(n.endswith(".part") for n in nombres) for nombres in visto)

    async def test_el_reintento_no_hereda_el_archivo_del_intento_fallido(self, tmp_path):
        """
        Si el primer intento deja restos, el segundo escribe encima de ellos y
        puede acabar con una mezcla de los dos.
        """
        path = str(tmp_path / "out.h5")
        estado = {"intento": 0}

        def nueva_lectura():
            estado["intento"] += 1
            if estado["intento"] == 1:
                return AsyncMock(side_effect=[b"corto", b""])
            return AsyncMock(side_effect=[b"contenido completo", b""])

        resp = _make_resp(200, "", content_length=18)
        resp.content.read = nueva_lectura()

        session = MagicMock()
        ctx = MagicMock()

        async def entrar(*a, **k):
            if estado["intento"] >= 1:
                resp.content.read = nueva_lectura()
            return resp

        ctx.__aenter__ = AsyncMock(side_effect=entrar)
        ctx.__aexit__ = AsyncMock(return_value=None)
        session.get = MagicMock(return_value=ctx)

        result = await download_file_async(
            session, "http://example.com/f.h5", path, max_retries=3, delay=0)

        assert result == path
        assert (tmp_path / "out.h5").read_bytes() == b"contenido completo"
        assert list(tmp_path.glob("*.part")) == []
