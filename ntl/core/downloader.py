"""
Descarga de archivos Black Marble diarios (VNP46A2, VJ146A2) desde LAADS.

Aquí es donde la distinción entre síncrono y asíncrono es real: el resto del
procesamiento no la necesita. Las funciones bloqueantes sirven para una consulta
puntual; las asíncronas, con sufijo `_async`, para los lotes.

Contrato de las funciones asíncronas: devuelven None cuando el gránulo no existe
o se agotan los reintentos, y lanzan `CredencialesInvalidas` cuando LAADS
rechaza el token (reintentar no sirve y lo demás fallará igual).
"""
import asyncio
import contextlib
import email.utils
import os
import random
import time
import uuid
import weakref
from collections import OrderedDict
from datetime import date, timedelta
from typing import NamedTuple
from urllib.parse import urljoin, urlparse

import aiohttp
import requests

from . import config
from .config import CHUNK_SIZE, HEADERS
from .errores import CredencialesInvalidas
from .limitador import obtener_limitador
from .productos import obtener_producto

FIRMA_HDF5 = b"\x89HDF\r\n\x1a\n"

# 1 MiB: con 8 KB cada chunk era una vuelta al loop y una escritura por cada
# 8 KB, es decir ~2 000 por gránulo de 16 MB.
CHUNK_ASYNC = 1024 * 1024

# Sin tope total: un gránulo en un enlace lento o congestionado tarda más de
# cualquier total razonable y se mataba a mitad de camino. Lo que sí se vigila es
# que la conexión no se quede muda (sock_read) o no se establezca (connect).
TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=60)

MAX_REDIRECTS = 10
_REDIRECCIONES = (301, 302, 303, 307, 308)
# Estados transitorios: el mismo pedido puede salir bien más tarde.
_REINTENTABLES = (408, 425, 429)
# El token solo viaja a NASA. LAADS redirige a Earthdata (urs.earthdata.nasa.gov)
# y a almacenamiento en la nube; a cualquier otro host el Bearer no se reenvía.
_SUFIJOS_NASA = (".nasa.gov",)
# Cuántos listados de día se recuerdan por loop; son ~30 KB cada uno.
_MAX_LISTADOS = 128


class _Reintentar(Exception):
    """Fallo transitorio. `espera` es la que pidió el servidor (Retry-After), si la pidió."""

    def __init__(self, motivo, espera=None, penalizar=False):
        super().__init__(motivo)
        self.motivo = motivo
        self.espera = espera
        self.penalizar = penalizar


class _Definitivo(Exception):
    """Fallo que reintentar no arregla (404, 410...): el resultado es None."""


# Se pueden parchear en pruebas sin tocar el reloj de todo asyncio.
_dormir = asyncio.sleep
_aleatorio = random.uniform


class _Entrada(NamedTuple):
    """Un archivo del listado: dónde bajarlo, cuánto debe pesar y cuándo se produjo."""
    url: str
    size: int | None
    mtime: float


# Un 404 en el listado es definitivo y se recuerda como "sin listado". Distinto
# de None (fallo transitorio o día sin publicar), que nunca se recuerda.
_SIN_LISTADO: dict = {}


def _entradas_del_listado(datos) -> dict:
    """
    nombre -> (downloadsLink, size, mtime) de los .h5 de un listado JSON de LAADS.

    Tolerante: la raíz puede ser `{"content": [...]}` o directamente una lista, y
    se saltan las entradas sin nombre o sin enlace en vez de fallar. CPU pura
    pero de unos cientos de entradas: no hace falta hilo.
    """
    if isinstance(datos, dict):
        datos = datos.get("content", [])
    if not isinstance(datos, list):
        return {}
    entradas = {}
    for e in datos:
        if not isinstance(e, dict):
            continue
        nombre, enlace = e.get("name"), e.get("downloadsLink")
        if not nombre or not enlace or not str(nombre).endswith(".h5"):
            continue
        if e.get("resourceType", "File") != "File":
            continue
        size = e.get("size")
        mtime = e.get("mtime")
        entradas[nombre] = _Entrada(
            str(enlace).strip(),
            int(size) if isinstance(size, (int, float)) and size >= 0 else None,
            float(mtime) if isinstance(mtime, (int, float)) else 0.0,
        )
    return entradas


def _elegir_entrada(entradas, cuadrante, producto):
    """
    La entrada del cuadrante, o None.

    Coincidencia exacta por el token `.{cuadrante}.` y por el prefijo del
    producto: `cuadrante in href` aceptaba "h08v07" dentro de cualquier cadena y
    no distinguía un VJ146A2 de un VNP46A2 si el listado trajera ambos. Un
    reproceso publica otro archivo para el mismo cuadrante (cambia la marca de
    producción del nombre): se toma el de `mtime` más reciente.
    """
    prefijo = f"{producto.nombre}."
    token = f".{cuadrante}."
    candidatas = [(nombre, e) for nombre, e in entradas.items()
                  if nombre.startswith(prefijo) and token in nombre]
    if not candidatas:
        return None
    # El nombre desempata (lleva la marca de producción) si los mtime coinciden.
    return max(candidatas, key=lambda ne: (ne[1].mtime, ne[0]))[1]


def _fecha_de(year, day) -> date:
    return date(int(year), 1, 1) + timedelta(days=int(day) - 1)


def _motivo_fuera_de_rango(producto, year, day):
    return producto.motivo_fuera_de_rango(_fecha_de(year, day))


def find_file(year, day, cuadrante, producto=None):
    """Busca el .h5 de un año, día y cuadrante en el listado JSON de LAADS. Bloqueante."""
    producto = obtener_producto(producto)
    motivo = _motivo_fuera_de_rango(producto, year, day)
    if motivo:
        print(f"{motivo}; no se pide a NASA")
        return None

    url = producto.url_listado(year, day)
    response = requests.get(url, headers=HEADERS, timeout=(30, 60))

    if response.status_code != 200:
        print(f"Error al acceder a {url}")
        return None

    try:
        entradas = _entradas_del_listado(response.json())
    except ValueError:
        print(f"El listado de {url} no es JSON")
        return None
    entrada = _elegir_entrada(entradas, cuadrante, producto)
    if entrada:
        return entrada.url

    print(f"No se encontró archivo {producto.nombre} para {cuadrante} en {year}-{day}")
    return None


def _ruta_parcial(save_path) -> str:
    """`.part` único por descarga: dos descargas del mismo gránulo no se pisan."""
    return f"{save_path}.{os.getpid()}.{uuid.uuid4().hex}.part"


def download_file(file_url: str, save_path: str) -> str:
    """
    Descarga un archivo a save_path y verifica que sea HDF5. Bloqueante.

    Escribe a un `.part` y renombra al final: en la ruta final solo aparece un
    archivo entero y válido.
    """
    parcial = _ruta_parcial(save_path)
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(file_url, headers=HEADERS, stream=True, timeout=(30, 60))

        if response.status_code != 200:
            print(f"Error al descargar: {file_url} (Status: {response.status_code})")
            return None

        with open(parcial, "wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                file.write(chunk)

        if not is_valid_hdf5_file(parcial):
            print(f"El archivo descargado no es un archivo HDF5 válido: {save_path}")
            _borrar(parcial)
            return None

        os.replace(parcial, save_path)
        print(f"Archivo descargado: {save_path}")
        return save_path

    except Exception as e:
        print(f"Error durante la descarga: {e}")
        _borrar(parcial)
        return None


def is_valid_hdf5_file(file_path: str) -> bool:
    """
    Verifica que el archivo sea HDF5 y se pueda abrir.

    LAADS responde con una página HTML de error y código 200 cuando el token no
    es válido, así que revisar solo el status no basta.
    """
    import h5py

    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"El archivo no existe o está vacío: {file_path}")
            return False

        with open(file_path, "rb") as f:
            header = f.read(8)

        if header != b"\x89HDF\r\n\x1a\n":
            with open(file_path, "rb") as f:
                inicio = f.read(100).lower()
            if b"<html" in inicio or b"<!doctype" in inicio:
                print("Se detectó contenido HTML (página de error) en lugar de archivo HDF5")
            else:
                print(f"El archivo no tiene la firma HDF5 correcta. Primeros bytes: {header}")
            return False

        with h5py.File(file_path, "r") as f:
            if len(list(f.keys())) == 0:
                print("El archivo HDF5 está vacío (sin grupos)")
                return False
        return True

    except Exception as e:
        print(f"Error verificando archivo HDF5: {e}")
        return False


def _borrar(ruta):
    """Quita un archivo a medias sin que su ausencia sea un problema."""
    try:
        if os.path.exists(ruta):
            os.remove(ruta)
    except OSError as e:
        print(f"No se pudo borrar {ruta}: {e}")


def _cabeceras(url: str) -> dict:
    """
    Cabeceras para `url`. El Bearer solo va a hosts de NASA y por https.

    aiohttp ya quita Authorization en un redirect a otro origen, pero se siguen
    a mano (LAADS necesita el token tras redirigir a Earthdata), así que la
    decisión de a quién se le manda el token es de este código.
    `Accept-Encoding: identity` evita que el servidor comprima un HDF5, que ya
    viene comprimido, y que Content-Length deje de coincidir con lo recibido.
    """
    cabeceras = {"Accept-Encoding": "identity"}
    partes = urlparse(url)
    host = (partes.hostname or "").lower()
    if partes.scheme == "https" and any(host.endswith(suf) for suf in _SUFIJOS_NASA):
        cabeceras.update(HEADERS)
    else:
        cabeceras.update({k: v for k, v in HEADERS.items() if k.lower() != "authorization"})
    return cabeceras


@contextlib.asynccontextmanager
async def _get(session, url):
    """GET que sigue los redirects a mano, decidiendo en cada salto qué cabeceras enviar."""
    actual = url
    for _ in range(MAX_REDIRECTS):
        async with session.get(actual, headers=_cabeceras(actual), timeout=TIMEOUT,
                               allow_redirects=False) as resp:
            if resp.status in _REDIRECCIONES:
                location = resp.headers.get("Location")
                if location:
                    actual = urljoin(actual, location)
                    continue
            yield resp
            return
    raise aiohttp.TooManyRedirects(None, (), message=f"demasiados redirects: {url}")


def _espera_retry_after(valor):
    """Segundos de un Retry-After (entero o fecha HTTP), o None si no viene o no se entiende."""
    if not valor:
        return None
    valor = valor.strip()
    try:
        return max(0.0, float(valor))
    except ValueError:
        pass
    try:
        fecha = email.utils.parsedate_to_datetime(valor)
        return max(0.0, fecha.timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def _evaluar_status(resp, url):
    """Traduce un status distinto de 200 en la excepción que decide qué hacer."""
    status = resp.status
    if status == 200:
        return
    if status in (401, 403):
        raise CredencialesInvalidas(
            f"LAADS respondió {status} para {url}: revisa NASA_API_TOKEN")
    if status in _REINTENTABLES or 500 <= status < 600:
        raise _Reintentar(
            f"HTTP {status}",
            espera=_espera_retry_after(resp.headers.get("Retry-After")),
            # Un 429/503 es el servidor diciendo "estás yendo muy rápido": hay
            # que frenar a todos, no solo a quien lo recibió.
            penalizar=status in (429, 503),
        )
    # 400, 404, 410 y el resto de 4xx: el mismo pedido dará lo mismo.
    raise _Definitivo(f"HTTP {status}")


async def _con_reintentos(intento, descripcion, max_intentos, base, pesada=False):
    """
    Ejecuta `intento()` con backoff exponencial y full jitter.

    Devuelve su resultado, o None si es un fallo definitivo o se agotan los
    intentos. `CredencialesInvalidas` y todo lo que no sea red (ENOSPC, bugs)
    se propagan: reintentarlos escondería el problema.
    """
    limitador = obtener_limitador()
    tope = config.BACKOFF_TOPE
    for n in range(1, max_intentos + 1):
        await limitador.esperar_turno()
        try:
            if pesada:
                async with limitador.descarga():
                    return await intento()
            return await intento()
        except _Definitivo as e:
            print(f"{descripcion}: {e}; no se reintenta")
            return None
        except _Reintentar as e:
            motivo, pedida, penalizar = e.motivo, e.espera, e.penalizar
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
            motivo, pedida, penalizar = f"{type(e).__name__}: {e}", None, False

        if n == max_intentos:
            print(f"{descripcion}: {motivo}; se agotaron los {max_intentos} intentos")
            return None
        if pedida is not None:
            espera = min(pedida, tope)
        else:
            espera = _aleatorio(0, min(tope, base * 2 ** (n - 1)))
        if penalizar:
            limitador.penalizar(espera)
        print(f"{descripcion}: {motivo} (intento {n}/{max_intentos}); "
              f"reintentando en {espera:.1f} s")
        await _dormir(espera)
    return None


# Listados de día ya pedidos, por event loop (un future no sirve en otro loop).
_listados: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
# url de descarga -> tamaño exacto anunciado por el listado, por loop. Lo consulta
# download_file_async: el listado dice cuánto pesa el archivo aunque el servidor
# no mande Content-Length (o lo mande de una respuesta comprimida).
_tamanos: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_MAX_TAMANOS = 8192


def _recordar_tamanos(entradas):
    tamanos = _tamanos.setdefault(asyncio.get_running_loop(), OrderedDict())
    for e in entradas.values():
        if e.size is not None:
            tamanos[e.url] = e.size
            tamanos.move_to_end(e.url)
    while len(tamanos) > _MAX_TAMANOS:
        tamanos.popitem(last=False)


def _tamano_esperado(url):
    try:
        return _tamanos.get(asyncio.get_running_loop(), {}).get(url)
    except RuntimeError:
        return None


async def _pedir_listado(session, url, max_intentos, base):
    async def intento():
        try:
            async with _get(session, url) as resp:
                _evaluar_status(resp, url)
                return await resp.json(content_type=None)
        except _Definitivo as e:
            # 404 y compañía: el mismo pedido dará lo mismo, y se recuerda.
            print(f"Listado {url}: {e}; no se reintenta")
            return _SIN_LISTADO

    datos = await _con_reintentos(intento, f"Listado {url}", max_intentos, base)
    if datos is None:
        return None
    if datos is _SIN_LISTADO:
        return _SIN_LISTADO
    entradas = _entradas_del_listado(datos)
    if not entradas:
        # Lista vacía: el día todavía no se publica. Cambia con las horas, así
        # que no se recuerda (None).
        print(f"Listado {url}: vacío (día sin publicar)")
        return None
    _recordar_tamanos(entradas)
    return entradas


def _olvidar_si_fallo(cache, clave):
    def _limpiar(fut):
        if fut.cancelled():
            fallo = True
        else:
            # Leer la excepción la da por atendida (si no, asyncio protesta).
            fallo = fut.exception() is not None or fut.result() is None
        # Un fallo no se recuerda: ser transitorio es lo normal, y «todavía no
        # publicado» cambia con las horas. Un 404 sí (_SIN_LISTADO no es None).
        if fallo and cache.get(clave) is fut:
            del cache[clave]
    return _limpiar


async def _listado_del_dia(session, producto, year, day, max_intentos=None, base=None):
    """
    Archivos .h5 del día `nombre -> (downloadsLink, size, mtime)`, pedidos una sola vez.

    Devuelve None si no se pudo listar o el día no está publicado (lista vacía),
    y un dict vacío si LAADS respondió 404. El listado es el mismo para todos los
    cuadrantes de un día: se comparte el future (single-flight) en lugar de
    guardar solo el resultado, para que quienes llegan mientras el primero sigue
    esperando se sumen a su petición. La clave incluye el producto: el mismo día
    tiene un listado por producto.
    """
    loop = asyncio.get_running_loop()
    cache = _listados.setdefault(loop, OrderedDict())
    clave = (producto.nombre, str(year), str(day))
    fut = cache.get(clave)
    if fut is None:
        url = producto.url_listado(year, day)
        fut = asyncio.ensure_future(_pedir_listado(
            session, url,
            max_intentos if max_intentos is not None else config.MAX_INTENTOS,
            base if base is not None else config.BACKOFF_BASE))
        cache[clave] = fut
        fut.add_done_callback(_olvidar_si_fallo(cache, clave))
        while len(cache) > _MAX_LISTADOS:
            cache.popitem(last=False)
    else:
        cache.move_to_end(clave)
    # shield: que cancelar a un llamador no mate la petición que comparten otros.
    return await asyncio.shield(fut)


async def find_file_async(session, year, day, cuadrante, producto=None):
    """
    URL del .h5 del cuadrante ese día, o None si no existe o no se pudo listar.

    Si el día cae fuera de la vigencia del producto devuelve None sin pedir nada.
    """
    producto = obtener_producto(producto)
    motivo = _motivo_fuera_de_rango(producto, year, day)
    if motivo:
        print(f"{motivo}; no se pide a NASA")
        return None
    entradas = await _listado_del_dia(session, producto, year, day)
    if not entradas:
        return None
    entrada = _elegir_entrada(entradas, cuadrante, producto)
    if entrada is None:
        print(f"No se encontró archivo {producto.nombre} para el cuadrante "
              f"{cuadrante} en {year}-{day}")
        return None
    return entrada.url


async def _escribir_y_validar(resp, parcial, url, esperado_listado=None):
    """
    Vuelca la respuesta a `parcial` y comprueba firma y tamaño. Devuelve los bytes escritos.

    El tamaño del listado, si se conoce, manda sobre Content-Length: es el del
    archivo entero y no depende de que el servidor lo anuncie bien.
    """
    esperado = resp.headers.get("Content-Length")
    esperado = int(esperado) if esperado and esperado.isdigit() else None
    # aiohttp descomprime solo: si hubo Content-Encoding, Content-Length cuenta
    # los bytes comprimidos y compararlo descartaría un archivo bueno.
    if resp.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
        esperado = None
    if esperado_listado is not None:
        esperado = esperado_listado

    total = 0
    cabeza = b""
    verificada = False
    with open(parcial, "wb") as f:
        while True:
            chunk = await resp.content.read(CHUNK_ASYNC)
            if not chunk:
                break
            if not verificada:
                # Se juzga con el primer chunk, sin esperar a bajar cientos de
                # MB de una página de login.
                cabeza = (cabeza + chunk)[:512]
                if len(cabeza) >= 16:
                    _revisar_firma(cabeza, url)
                    verificada = True
            # La escritura bloquea (disco): se hace en un hilo para no frenar
            # las demás descargas que comparten el loop.
            await asyncio.to_thread(f.write, chunk)
            total += len(chunk)

    if not verificada:
        _revisar_firma(cabeza, url)  # cuerpo vacío o de menos de 16 bytes
    if esperado is not None and total != esperado:
        raise _Reintentar(f"descarga incompleta: {total} de {esperado} bytes")
    return total


def _revisar_firma(cabeza: bytes, url: str):
    if cabeza.startswith(FIRMA_HDF5):
        return
    inicio = cabeza[:100].lower()
    if b"<html" in inicio or b"<!doctype" in inicio:
        # LAADS contesta 200 con la página de login cuando el token no vale.
        raise CredencialesInvalidas(
            f"LAADS devolvió una página HTML en lugar del HDF5 ({url}): "
            "el token NASA_API_TOKEN es inválido o expiró")
    raise _Reintentar("el contenido no tiene firma HDF5")


async def download_file_async(session, url, path, max_retries=None, delay=None):
    """
    Descarga `url` a `path`. Devuelve `path`, o None si no existe o se agotan los reintentos.

    `max_retries` y `delay` (base del backoff, en segundos) toman por omisión
    NTL_MAX_INTENTOS y NTL_BACKOFF_BASE. Lanza `CredencialesInvalidas` ante un
    401/403 o una página HTML con código 200.

    Escribe a un `.part` único (pid + uuid) y lo renombra a `path` solo si el
    tamaño cuadra y la firma es HDF5: una conexión cortada o dos descargas
    concurrentes del mismo gránulo no pueden dejar ahí un archivo a medias ni
    pisarse entre sí. Los redirects se siguen a mano porque aiohttp quita
    Authorization al cambiar de host y LAADS pasa por Earthdata; el token solo
    se reenvía a hosts de NASA por https.
    """
    max_intentos = max_retries if max_retries is not None else config.MAX_INTENTOS
    base = delay if delay is not None else config.BACKOFF_BASE
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Tamaño exacto del listado si esta URL salió de uno (None si no).
    esperado_listado = _tamano_esperado(url)

    async def intento():
        parcial = _ruta_parcial(path)
        try:
            async with _get(session, url) as resp:
                _evaluar_status(resp, url)
                print(f"Descargando: {url}")
                total = await _escribir_y_validar(resp, parcial, url, esperado_listado)
            # El renombrado es atómico: en la ruta final solo aparece un
            # archivo entero, nunca uno a medio escribir.
            os.replace(parcial, path)
            print(f"Descarga completada: {path} ({total} bytes)")
            return path
        except BaseException:
            _borrar(parcial)
            raise

    return await _con_reintentos(intento, f"Descarga {url}", max_intentos, base, pesada=True)
