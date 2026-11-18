"""
Descarga parcial de un gránulo: solo las filas que usan los municipios.

Un cuadrante Black Marble es una retícula de 2400×2400 guardada en franjas de
100 filas comprimidas por separado. Las alcaldías de la CDMX y Oaxaca caen en 5
de las 24 franjas de h08v07, y el procesamiento solo usa 4 de las 7 capas. Bajar
el archivo entero era traer ~19 MB para usar ~4.4.

LAADS sirve rangos de bytes (responde 206), así que el HDF5 se abre en remoto:
h5py lee a través de `_ArchivoRemoto`, que pide por HTTP solo los bloques que
toca, y de ahí se escribe a disco un HDF5 **delgado** con la misma estructura
—mismos grupos, capas, atributos, forma y metadatos de georreferencia— pero con
datos solo en las filas pedidas. El resto de la retícula queda con el valor de
relleno del propio archivo. Todo lo que viene después (lectura, verificación de
georreferencia, métricas, recorte) no cambia: abre un HDF5 local como siempre.

Las filas que se piden cubren el **bounding box completo** de cada municipio en
cada cuadrante, no solo sus píxeles: el recorte que se guarda como
`Matriz_de_radianza` abarca ese rectángulo entero, y con filas de menos saldría
distinto del que da el archivo completo.

El archivo delgado lleva el atributo raíz `ntl_filas_parciales`: no es un
gránulo completo y no debe usarse como tal fuera de este flujo.

Cuándo no se hace (y se baja el archivo entero, como antes): si el servidor
ignora el Range y responde 200, o si el archivo no tiene la forma esperada.
Eso lo decide `_NoParcial`, que el llamador traduce a descarga completa.
"""
from __future__ import annotations

import io
import os
import random
import threading
import uuid
from typing import Callable, Iterable
from urllib.parse import urljoin

import h5py
import requests

from . import config
from .config import find_image_path
from .errores import CredencialesInvalidas, DescargaFallida

ATRIBUTO_PARCIAL = "ntl_filas_parciales"
_REDIRECCIONES = (301, 302, 303, 307, 308)
_REINTENTABLES = (408, 425, 429)
# (conexión, lectura). La lectura corta a propósito: cada petición es de pocos
# cientos de KB, y una que se queda colgada detiene todo el proceso, porque h5py
# lee bajo un candado global. Con LAADS lento, un timeout de 60 s dejaba
# procesos enteros quietos; mejor cortar y reintentar.
_TIMEOUT = (15, 20)
# Las esperas entre reintentos ocurren dentro de una lectura de h5py, es decir,
# con su candado tomado: mientras se duerme, nada más de h5py avanza en el
# proceso. Por eso el backoff propio se acota aquí. Un Retry-After del servidor
# se respeta entero (hasta BACKOFF_TOPE) y además se pasa al limitador global
# para que frene a todos.
_TOPE_ESPERA_EN_HILO = 10.0


class _NoParcial(Exception):
    """La lectura parcial no aplica a este archivo: hay que bajarlo entero."""


class _NoExiste(Exception):
    """El gránulo ya no está en el servidor (404/410)."""


class _Cancelado(Exception):
    """Quien pidió la descarga la canceló: no se pide nada más ni se deja archivo."""


class _RangoEquivocado(requests.RequestException):
    """El 206 no trae el rango pedido (inicio o largo distinto): se reintenta."""


class _ArchivoRemoto(io.RawIOBase):
    """
    Archivo de solo lectura sobre HTTP, por bloques y con caché.

    Cada lectura de h5py se traduce a los bloques que toca; los que faltan y son
    contiguos se piden en una sola petición. Un bloque ya traído no se vuelve a
    pedir. Lleva la cuenta de peticiones y bytes, que es lo que se quiere bajar.
    """

    def __init__(self, url: str, sesion: requests.Session, bloque: int,
                 max_intentos: int, base: float, tope: float,
                 cancelado: threading.Event | None = None,
                 penalizar: Callable[[float], None] | None = None):
        super().__init__()
        self._cancelado = cancelado or threading.Event()
        self._penalizar = penalizar
        self.url = url
        self._sesion = sesion
        self._bloque = bloque
        self._max_intentos = max_intentos
        self._base = base
        self._tope = tope
        self._bloques: dict[int, bytes] = {}
        self._pos = 0
        self.peticiones = 0
        self.bytes_recibidos = 0
        # h5py envuelve lo que lance un archivo de Python en su propio error y
        # se pierde el tipo; se guarda aquí para relanzarlo tal cual
        # (CredencialesInvalidas tiene que llegar como CredencialesInvalidas).
        self.error: BaseException | None = None
        # El primer bloque trae, en Content-Range, el tamaño total del archivo.
        self.tamano = None
        self._traer(0, 0)
        if self.tamano is None:
            raise _NoParcial("el servidor no informó el tamaño del archivo")

    # --- interfaz de archivo -------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, desplazamiento, desde=io.SEEK_SET):
        if desde == io.SEEK_SET:
            self._pos = desplazamiento
        elif desde == io.SEEK_CUR:
            self._pos += desplazamiento
        else:
            self._pos = self.tamano + desplazamiento
        return self._pos

    def readinto(self, destino):
        try:
            return self._leer_en(destino)
        except BaseException as e:
            self.error = e
            raise

    def _leer_en(self, destino):
        n = min(len(destino), self.tamano - self._pos)
        if n <= 0:
            return 0
        primero, ultimo = self._pos // self._bloque, (self._pos + n - 1) // self._bloque
        faltan = [k for k in range(primero, ultimo + 1) if k not in self._bloques]
        # Bloques faltantes contiguos → una sola petición.
        while faltan:
            inicio = fin = faltan.pop(0)
            while faltan and faltan[0] == fin + 1:
                fin = faltan.pop(0)
            self._traer(inicio, fin)
        datos = b"".join(self._bloques[k] for k in range(primero, ultimo + 1))
        corte = self._pos - primero * self._bloque
        destino[:n] = datos[corte:corte + n]
        self._pos += n
        return n

    # --- red -----------------------------------------------------------------
    def _traer(self, primer_bloque: int, ultimo_bloque: int):
        a = primer_bloque * self._bloque
        b = (ultimo_bloque + 1) * self._bloque - 1
        if self.tamano is not None:
            b = min(b, self.tamano - 1)
        cuerpo = self._pedir_rango(a, b)
        for k in range(primer_bloque, ultimo_bloque + 1):
            ini = (k - primer_bloque) * self._bloque
            self._bloques[k] = cuerpo[ini:ini + self._bloque]

    def _pedir_rango(self, a: int, b: int) -> bytes:
        """GET con Range y reintentos. Mismas reglas que la descarga completa."""
        # Import tardío: downloader importa este módulo.
        from .downloader import _cabeceras, _espera_retry_after

        rechazos_seguidos = 0
        for n in range(1, self._max_intentos + 1):
            if self._cancelado.is_set():
                raise _Cancelado(self.url)
            espera_pedida = None
            penalizar = False
            try:
                actual = self.url
                for _ in range(10):
                    cabeceras = dict(_cabeceras(actual), Range=f"bytes={a}-{b}")
                    resp = self._sesion.get(actual, headers=cabeceras, timeout=_TIMEOUT,
                                            allow_redirects=False, stream=False)
                    if resp.status_code in _REDIRECCIONES and resp.headers.get("Location"):
                        actual = urljoin(actual, resp.headers["Location"])
                        continue
                    break
                self.peticiones += 1
                status = resp.status_code
                if status == 206:
                    cuerpo = resp.content
                    esperado = b - a + 1
                    inicio, total = _content_range(resp.headers.get("Content-Range"))
                    if inicio is not None and inicio != a:
                        # Un proxy o caché que desplace el rango corrompería la
                        # lectura en silencio: los bytes irían a otra posición.
                        raise _RangoEquivocado(
                            f"se pidió el byte {a} y llegó desde el {inicio}")
                    if total is not None:
                        self.tamano = total
                        esperado = min(b, total - 1) - a + 1
                    if len(cuerpo) != esperado:
                        raise _RangoEquivocado(
                            f"rango incompleto: {len(cuerpo)} de {esperado} bytes")
                    self.bytes_recibidos += len(cuerpo)
                    return cuerpo
                if status == 200:
                    # Ignoró el Range: seguir sería bajar el archivo entero por
                    # trozos repetidos. Mejor la descarga completa de siempre.
                    resp.close()
                    raise _NoParcial("el servidor no acepta rangos (respondió 200)")
                if status in (401, 403):
                    # Con decenas de peticiones por gránulo, un rechazo aislado
                    # de LAADS no basta para declarar el token inválido (ya se
                    # vieron 401 pasajeros con un token vigente). Dos seguidos sí.
                    rechazos_seguidos += 1
                    if rechazos_seguidos >= 2:
                        raise CredencialesInvalidas(
                            f"LAADS respondió {status} para {self.url}: revisa NASA_API_TOKEN")
                    motivo = f"HTTP {status}"
                elif status in (404, 410):
                    raise _NoExiste(f"HTTP {status}")
                elif status in _REINTENTABLES or 500 <= status < 600:
                    rechazos_seguidos = 0
                    motivo = f"HTTP {status}"
                    espera_pedida = _espera_retry_after(resp.headers.get("Retry-After"))
                    # 429/503: el servidor dice «vas muy rápido». Igual que en la
                    # descarga completa, hay que frenar a todos, no solo a este hilo.
                    penalizar = status in (429, 503)
                else:
                    raise DescargaFallida(f"Rango {a}-{b} de {self.url}: HTTP {status}")
            except requests.RequestException as e:
                # Toda falla de red es transitoria aquí, incluido un cuerpo
                # cortado a media respuesta (ChunkedEncodingError), que no es
                # ConnectionError y antes escapaba sin reintentarse.
                rechazos_seguidos = 0
                motivo = f"{type(e).__name__}: {e}"

            if n == self._max_intentos:
                raise DescargaFallida(
                    f"Rango {a}-{b} de {self.url}: {motivo}; se agotaron los "
                    f"{self._max_intentos} intentos")
            espera = (min(espera_pedida, self._tope) if espera_pedida is not None
                      else random.uniform(0, min(self._tope, self._base * 2 ** (n - 1))))
            if penalizar and self._penalizar is not None:
                self._penalizar(espera)
            if espera_pedida is None:
                # El backoff propio sí se acota: es una conjetura nuestra.
                # Un Retry-After no: el servidor dijo cuánto, y reintentar antes
                # solo gasta intentos contra un 429 seguro.
                espera = min(espera, _TOPE_ESPERA_EN_HILO)
            print(f"Rango {a}-{b}: {motivo} (intento {n}/{self._max_intentos}); "
                  f"reintentando en {espera:.1f} s")
            # `wait` en vez de sleep: una cancelación corta la espera.
            if self._cancelado.wait(espera):
                raise _Cancelado(self.url)
        raise AssertionError("inalcanzable")


def _content_range(valor: str | None) -> tuple[int | None, int | None]:
    """`bytes 0-65535/19343964` → (0, 19343964). None en lo que no venga."""
    if not valor:
        return None, None
    inicio = total = None
    cuerpo = valor.strip()
    if cuerpo.lower().startswith("bytes"):
        cuerpo = cuerpo[5:].strip()
    rango, _, tot = cuerpo.partition("/")
    a = rango.split("-", 1)[0].strip()
    if a.isdigit():
        inicio = int(a)
    if tot.strip().isdigit():
        total = int(tot.strip())
    return inicio, total


def unir_intervalos(intervalos: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Intervalos semiabiertos [a, b) ordenados y sin solapes."""
    unidos: list[list[int]] = []
    for a, b in sorted(i for i in intervalos if i[0] < i[1]):
        if unidos and a <= unidos[-1][1]:
            unidos[-1][1] = max(unidos[-1][1], b)
        else:
            unidos.append([a, b])
    return [(a, b) for a, b in unidos]


def _copiar_atributos(origen, destino) -> None:
    for clave, valor in origen.attrs.items():
        try:
            destino.attrs[clave] = valor
        except (TypeError, ValueError, OSError):
            # Referencias a objetos (p. ej. DIMENSION_LIST) no se pueden copiar
            # entre archivos tal cual; el procesamiento no las usa.
            pass


def escribir_delgado(origen: h5py.File, destino: str,
                     filas_para: Callable[[tuple[int, int]], list[tuple[int, int]]]) -> list:
    """
    Copia la estructura de `origen` a `destino` con datos solo en las filas pedidas.

    Las capas con la forma de la retícula se crean con el mismo tipo, troceado y
    valor de relleno, y solo se escriben las filas pedidas; todo lo demás
    (metadatos, coordenadas, escalares) se copia entero. Devuelve los intervalos
    de filas escritos.
    """
    forma = origen[find_image_path(origen)].shape
    if len(forma) != 2:
        raise _NoParcial(f"la capa principal no es 2D: {forma}")
    filas = unir_intervalos(filas_para(tuple(forma)))
    if not filas:
        raise _NoParcial("ningún municipio usa este cuadrante")

    with h5py.File(destino, "w") as salida:
        _copiar_atributos(origen, salida)
        salida.attrs[ATRIBUTO_PARCIAL] = [list(i) for i in filas]

        def copiar(nombre, objeto):
            if isinstance(objeto, h5py.Group):
                grupo = salida.require_group(nombre)
                _copiar_atributos(objeto, grupo)
                return
            if not isinstance(objeto, h5py.Dataset):
                return
            if objeto.shape == forma:
                capa = salida.create_dataset(
                    nombre, shape=forma, dtype=objeto.dtype,
                    chunks=objeto.chunks or True, fillvalue=objeto.fillvalue,
                    compression="gzip", compression_opts=1)
                for a, b in filas:
                    capa[a:b, :] = objeto[a:b, :]
            else:
                capa = salida.create_dataset(nombre, data=objeto[()], dtype=objeto.dtype)
            _copiar_atributos(objeto, capa)

        origen.visititems(copiar)
    return filas


def bajar_franjas(url: str, path: str,
                  filas_para: Callable[[tuple[int, int]], list[tuple[int, int]]],
                  bloque: int | None = None,
                  cancelado: threading.Event | None = None,
                  penalizar: Callable[[float], None] | None = None) -> dict | None:
    """
    Baja de `url` solo las filas que pide `filas_para(forma)` y deja en `path`
    un HDF5 delgado (ver módulo). Bloqueante: el lote la corre en un hilo.

    Devuelve estadísticas {peticiones, bytes, tamano, filas}, o None si el
    gránulo no existe. Lanza `_NoParcial` si hay que bajarlo entero,
    `CredencialesInvalidas` y `DescargaFallida` como la descarga completa.
    Como ésta, escribe a un `.part` y solo renombra un archivo terminado.

    `cancelado`: si se activa, no se hace ninguna petición más, se corta la
    espera en curso y no queda archivo (lanza `_Cancelado`). `penalizar(s)`
    recibe la espera que pidió el servidor ante un 429/503.
    """
    cancelado = cancelado or threading.Event()
    bloque = bloque or config.BLOQUE_RANGO
    parcial = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.part"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with requests.Session() as sesion:
            try:
                remoto = _ArchivoRemoto(url, sesion, bloque, config.MAX_INTENTOS,
                                        config.BACKOFF_BASE, config.BACKOFF_TOPE,
                                        cancelado=cancelado, penalizar=penalizar)
            except _NoExiste:
                return None
            try:
                with h5py.File(remoto, "r") as origen:
                    filas = escribir_delgado(origen, parcial, filas_para)
            except Exception:
                if remoto.error is not None:
                    raise remoto.error from None
                raise
        if cancelado.is_set():
            # Ya terminó de bajar, pero quien lo pidió se fue: el directorio de
            # la corrida puede no existir, y nadie borraría este archivo.
            raise _Cancelado(url)
        os.replace(parcial, path)
        return {"peticiones": remoto.peticiones, "bytes": remoto.bytes_recibidos,
                "tamano": remoto.tamano, "filas": filas}
    except _NoExiste:
        return None
    finally:
        if os.path.exists(parcial):
            os.remove(parcial)
