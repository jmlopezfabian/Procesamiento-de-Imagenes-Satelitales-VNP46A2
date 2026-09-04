"""
Descarga de archivos VNP46A2 desde LAADS.

Aquí es donde la distinción entre síncrono y asíncrono es real: el resto del
procesamiento no la necesita. Las funciones bloqueantes sirven para una consulta
puntual; las asíncronas, con sufijo `_async`, para los lotes.
"""
import os
from urllib.parse import urljoin

import aiohttp
import asyncio
import requests
from bs4 import BeautifulSoup

from .config import BASE_URL, CHUNK_SIZE, HEADERS


def _resolver_enlace(href: str, url_base: str) -> str:
    """Normaliza el href de un listado de LAADS a una URL absoluta."""
    href = href.strip()
    if href.startswith("http"):
        return href.replace("\n", "").replace("\r", "").strip()
    return url_base + href


def find_file(year, day, cuadrante):
    """Busca el .h5 de un año, día y cuadrante. Bloqueante."""
    url = BASE_URL.format(year=year, day=day)
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Error al acceder a {url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.find_all("a"):
        filename = link.get("href")
        if filename and cuadrante in filename and filename.endswith(".h5"):
            return _resolver_enlace(filename, url)

    print(f"No se encontró archivo para {cuadrante} en {year}-{day}")
    return None


def download_file(file_url: str, save_path: str) -> str:
    """Descarga un archivo a save_path y verifica que sea HDF5. Bloqueante."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(file_url, headers=HEADERS, stream=True, timeout=30)

        if response.status_code != 200:
            print(f"Error al descargar: {file_url} (Status: {response.status_code})")
            return None

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                file.write(chunk)

        if not is_valid_hdf5_file(save_path):
            print(f"El archivo descargado no es un archivo HDF5 válido: {save_path}")
            os.remove(save_path)
            return None

        print(f"Archivo descargado: {save_path}")
        return save_path

    except Exception as e:
        print(f"Error durante la descarga: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
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


async def find_file_async(session, year, day, cuadrante):
    url = BASE_URL.format(year=year, day=day)
    async with session.get(url, headers=HEADERS) as resp:
        if resp.status != 200:
            print(f"Error al acceder a {url}")
            return None
        text = await resp.text()
        soup = BeautifulSoup(text, "html.parser")
        for link in soup.find_all("a"):
            filename = link.get("href")
            if filename and cuadrante in filename and filename.endswith(".h5"):
                # Limpiar el href de espacios y saltos de línea
                filename = filename.strip()
                
                # Verificar si el href ya es una URL completa
                if filename.startswith("http"):
                    # Reemplazar saltos de línea y espacios extra
                    filename = filename.replace('\n', '').replace('\r', '').strip()
                    return filename
                else:
                    # Si es solo el nombre del archivo, concatenar con la URL base
                    full_url = url + filename
                    return full_url
    print(f"No se encontró archivo para cuadrante {cuadrante} en {url}")
    return None

def _borrar(ruta):
    """Quita un archivo a medias sin que su ausencia sea un problema."""
    try:
        if os.path.exists(ruta):
            os.remove(ruta)
    except OSError as e:
        print(f"No se pudo borrar {ruta}: {e}")


async def download_file_async(session, url, path, max_retries=3, delay=2):
    """
    Descarga un archivo con sistema de retry.

    Sigue redirects manualmente preservando el header Authorization, ya que aiohttp
    lo elimina en redirects a otro host (p.ej. Earthdata) y LAADS requiere el token.

    Escribe a `path + ".part"` y renombra al terminar, después de comprobar el
    tamaño contra Content-Length. Antes se escribía directo sobre la ruta final:
    una conexión cortada dejaba ahí un gránulo truncado que el cache daba por
    bueno y servía al resto de municipios de esa fecha. h5py no lo lee, así que
    no producía métricas malas —eso se comprobó—, pero el municipio salía como
    si le faltara la imagen: un registro parcial en vez de un reintento. Y el
    archivo corrupto se quedaba, así que el reintento nunca llegaba.
    """
    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            timeout = aiohttp.ClientTimeout(total=300, connect=60)
            current_url = url
            max_redirects = 10

            for _ in range(max_redirects):
                async with session.get(
                    current_url, headers=HEADERS, timeout=timeout, allow_redirects=False
                ) as resp:
                    if resp.status in (301, 302, 303, 307, 308):
                        location = resp.headers.get("Location")
                        if not location:
                            break
                        current_url = (
                            location
                            if location.startswith("http")
                            else urljoin(current_url, location)
                        )
                        continue
                    if resp.status == 200:
                        print(f"Descargando: {url} (intento {attempt + 1}/{max_retries})")
                        esperado = resp.headers.get("Content-Length")
                        esperado = int(esperado) if esperado and esperado.isdigit() else None
                        parcial = path + ".part"
                        total_size = 0
                        try:
                            with open(parcial, "wb") as f:
                                while True:
                                    chunk = await resp.content.read(8192)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                                    total_size += len(chunk)
                        except BaseException:
                            _borrar(parcial)
                            raise

                        if esperado is not None and total_size != esperado:
                            # Incompleto. Se tira: dejarlo sería peor que no
                            # tenerlo, porque tiene toda la pinta de un gránulo.
                            _borrar(parcial)
                            print(f"Descarga incompleta de {url}: {total_size} de "
                                  f"{esperado} bytes (intento {attempt + 1}/{max_retries})")
                            break

                        # El renombrado es atómico: en la ruta final solo aparece
                        # un archivo entero, nunca uno a medio escribir.
                        os.replace(parcial, path)
                        print(f"Descarga completada: {path} ({total_size} bytes)")
                        return path
                    print(f"Fallo la descarga del archivo: {url} - Status: {resp.status} (intento {attempt + 1}/{max_retries})")
                    try:
                        error_text = await resp.text()
                        print(f"Respuesta del servidor: {error_text[:200]}")
                    except Exception:
                        pass
                    break

            if attempt < max_retries - 1:
                print(f"Reintentando en {delay} segundos...")
                await asyncio.sleep(delay)
                continue
            return None
                        
        except aiohttp.ClientError as e:
            print(f"Error de cliente HTTP al descargar {url}: {e} (intento {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"Reintentando en {delay} segundos...")
                await asyncio.sleep(delay)
                continue
            else:
                return None
        except asyncio.TimeoutError as e:
            print(f"Timeout al descargar {url}: {e} (intento {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"Reintentando en {delay} segundos...")
                await asyncio.sleep(delay)
                continue
            else:
                return None
        except RuntimeError as e:
            print(f"Error de runtime al descargar {url}: {e} (intento {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"Reintentando en {delay} segundos...")
                await asyncio.sleep(delay)
                continue
            else:
                return None
        except Exception as e:
            print(f"Error inesperado al descargar {url}: {e} (intento {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"Reintentando en {delay} segundos...")
                await asyncio.sleep(delay)
                continue
            else:
                return None
    
    return None