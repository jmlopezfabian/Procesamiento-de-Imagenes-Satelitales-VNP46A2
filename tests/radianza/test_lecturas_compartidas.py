"""
Un cuadrante se descomprime una vez por fecha, no una vez por municipio.

h08v07 sirve a las 16 alcaldías de la CDMX y a Oaxaca. Releerlo para cada una
era el 98% del tiempo de procesamiento de una fecha (2.41 s → 0.38 s medido con
un gránulo real de VNP46A2, mismos resultados byte a byte). Estas pruebas fijan
que la memoria no cambia el resultado, que nunca sirve una imagen vieja y que se
suelta cuando el archivo se borra.
"""
import os
import threading
from datetime import date
from unittest.mock import patch

import numpy as np
import pytest

import ntl.radianza.extraccion as extraccion
from ntl.geometria.mosaico import cobertura_por_cuadrante
from ntl.radianza.extraccion import (
    leer_cuadrante_compartido,
    olvidar_lectura,
    process_image_mosaico,
)
from tests.test_multicuadrante import FORMA, crear_cuadrante, cuadrado

CUADRANTE = "h08v07"
# Centro de h08v07 en la retícula sintética (lon -100..-90, lat 10..20).
CENTRO = (-95.0, 15.0)


@pytest.fixture(autouse=True)
def memoria_limpia():
    extraccion._LECTURAS.clear()
    yield
    extraccion._LECTURAS.clear()


def _municipios():
    """Tres «municipios» distintos dentro del mismo cuadrante."""
    return {
        f"m{i}": cobertura_por_cuadrante(cuadrado((CENTRO[0] + d, CENTRO[1]), 0.8), FORMA)
        for i, d in enumerate((-2.0, 0.0, 2.0))
    }


def _contar_lecturas():
    original = extraccion._leer_cuadrante
    llamadas = []

    def contando(ruta, cuadrante):
        llamadas.append(ruta)
        return original(ruta, cuadrante)

    return llamadas, patch.object(extraccion, "_leer_cuadrante", side_effect=contando)


def test_varios_municipios_leen_el_cuadrante_una_sola_vez(tmp_path):
    ruta = crear_cuadrante(tmp_path, CUADRANTE)
    llamadas, parche = _contar_lecturas()
    with parche:
        for nombre, piezas in _municipios().items():
            r = process_image_mosaico({CUADRANTE: ruta}, piezas, date(2024, 1, 1),
                                      nombre, delete_files=False)
            assert r is not None
    assert llamadas == [ruta]


def test_el_resultado_es_el_mismo_con_y_sin_memoria(tmp_path):
    ruta = crear_cuadrante(tmp_path, CUADRANTE)
    municipios = _municipios()

    def correr():
        return {
            n: process_image_mosaico({CUADRANTE: ruta}, p, date(2024, 1, 1), n,
                                     delete_files=False).model_dump(mode="json")
            for n, p in municipios.items()
        }

    con_memoria = correr()
    # Sin memoria: cada municipio vuelve a abrir el archivo, como antes.
    with patch.object(extraccion, "leer_cuadrante_compartido",
                      side_effect=extraccion._leer_cuadrante):
        sin_memoria = correr()
    assert con_memoria == sin_memoria


def test_un_archivo_reemplazado_se_vuelve_a_leer(tmp_path):
    """
    La clave es la identidad del archivo, no su nombre: la siguiente fecha
    descarga a otra ruta, pero aun si reusara la misma, una imagen nueva no puede
    recibir los valores memorizados de la anterior.
    """
    ruta = crear_cuadrante(tmp_path, CUADRANTE)
    antes = leer_cuadrante_compartido(ruta, CUADRANTE)[0].copy()
    mtime = os.stat(ruta).st_mtime_ns

    crear_cuadrante(tmp_path, CUADRANTE, relleno=True)  # mismo nombre, otro contenido
    # Que no dependa de la resolución de mtime del sistema de archivos.
    os.utime(ruta, ns=(mtime + 10**9, mtime + 10**9))
    despues = leer_cuadrante_compartido(ruta, CUADRANTE)[0]

    assert np.isfinite(antes).all()
    assert np.isnan(despues).all()


def test_las_capas_compartidas_no_se_pueden_modificar(tmp_path):
    """Las comparten varios municipios: escribir en una contaminaría a los demás."""
    ruta = crear_cuadrante(tmp_path, CUADRANTE)
    matriz, _ = leer_cuadrante_compartido(ruta, CUADRANTE)
    with pytest.raises(ValueError):
        matriz[0, 0] = 1.0


def test_olvidar_suelta_la_memoria(tmp_path):
    ruta = crear_cuadrante(tmp_path, CUADRANTE)
    leer_cuadrante_compartido(ruta, CUADRANTE)
    assert len(extraccion._LECTURAS) == 1
    olvidar_lectura(ruta)
    assert len(extraccion._LECTURAS) == 0


def test_borrar_el_archivo_al_procesar_tambien_lo_olvida(tmp_path):
    ruta = crear_cuadrante(tmp_path, CUADRANTE)
    piezas = next(iter(_municipios().values()))
    process_image_mosaico({CUADRANTE: ruta}, piezas, date(2024, 1, 1), "m",
                          delete_files=True)
    assert len(extraccion._LECTURAS) == 0


def test_la_memoria_tiene_tope(tmp_path, monkeypatch):
    monkeypatch.setattr(extraccion, "MAX_LECTURAS_EN_MEMORIA", 2)
    for i in range(4):
        d = tmp_path / str(i)
        d.mkdir()
        leer_cuadrante_compartido(crear_cuadrante(d, CUADRANTE), CUADRANTE)
    assert len(extraccion._LECTURAS) == 2


def test_un_archivo_que_no_existe_no_se_memoriza(tmp_path):
    assert leer_cuadrante_compartido(str(tmp_path / "no.h5"), CUADRANTE) is None
    assert len(extraccion._LECTURAS) == 0


def test_el_lote_suelta_la_lectura_al_borrar_el_cuadrante(tmp_path):
    """`SatelliteImagesAsync` borra por contador; con el archivo se va la imagen."""
    from ntl.radianza.lotes import SatelliteImagesAsync

    ruta = crear_cuadrante(tmp_path, CUADRANTE)
    leer_cuadrante_compartido(ruta, CUADRANTE)
    sat = SatelliteImagesAsync.__new__(SatelliteImagesAsync)
    sat.cache_h5_files, sat._vuelo = {"k": ruta}, {}
    sat._borrar_cuadrante("k", ruta)
    assert len(extraccion._LECTURAS) == 0


def test_una_lectura_fallida_no_se_memoriza(tmp_path):
    """
    Un fallo puede ser pasajero (EMFILE, EIO). Si se memorizaba, los demás
    municipios de la fecha ya no reintentaban y salían sin ese cuadrante.
    """
    ruta = crear_cuadrante(tmp_path, CUADRANTE)
    original = extraccion._leer_cuadrante
    intentos = []

    def falla_una_vez(r, c):
        intentos.append(r)
        return None if len(intentos) == 1 else original(r, c)

    with patch.object(extraccion, "_leer_cuadrante", side_effect=falla_una_vez):
        assert leer_cuadrante_compartido(ruta, CUADRANTE) is None
        assert leer_cuadrante_compartido(ruta, CUADRANTE) is not None
    assert len(intentos) == 2


def test_olvidar_durante_una_lectura_no_deja_la_entrada(tmp_path):
    """
    Al abortar una fecha, su hilo puede seguir leyendo mientras el lote ya
    olvidó y borró el archivo. Al terminar no debe guardar la lectura: serían
    ~140 MB de un archivo que ya no existe.
    """
    ruta = crear_cuadrante(tmp_path, CUADRANTE)
    original = extraccion._leer_cuadrante
    leyendo, seguir = threading.Event(), threading.Event()

    def lenta(r, c):
        leyendo.set()
        seguir.wait(5)
        return original(r, c)

    with patch.object(extraccion, "_leer_cuadrante", side_effect=lenta):
        hilo = threading.Thread(target=leer_cuadrante_compartido, args=(ruta, CUADRANTE))
        hilo.start()
        leyendo.wait(5)
        olvidar_lectura(ruta)
        seguir.set()
        hilo.join(5)
    assert len(extraccion._LECTURAS) == 0
    # Y la contabilidad de olvidos no se queda con nada al terminar.
    assert extraccion._OLVIDOS == {} and extraccion._EN_VUELO == {}


def test_olvidar_sin_lecturas_en_curso_no_acumula(tmp_path):
    for i in range(5):
        olvidar_lectura(str(tmp_path / f"{i}.h5"))
    assert extraccion._OLVIDOS == {}
