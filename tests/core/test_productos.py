"""Registro de productos: valores por omisión, validación y vigencia."""
from datetime import date

import pytest

from ntl.core import config
from ntl.core.productos import PRODUCTOS, Producto, obtener_producto, producto_por_omision


@pytest.fixture(autouse=True)
def sin_env(monkeypatch):
    monkeypatch.delenv("NTL_PRODUCTO", raising=False)


def test_registro():
    assert set(PRODUCTOS) == {"VNP46A2", "VJ146A2", "VJ246A2"}
    vnp, vj, vj2 = PRODUCTOS["VNP46A2"], PRODUCTOS["VJ146A2"], PRODUCTOS["VJ246A2"]
    assert (vnp.inicio, vnp.fin, vnp.coleccion) == (date(2012, 1, 19), date(2026, 11, 2), 5200)
    assert (vj.inicio, vj.fin, vj.plataforma) == (date(2018, 1, 5), None, "JPSS-1")
    assert (vj2.inicio, vj2.fin, vj2.plataforma, vj2.satelite, vj2.coleccion) == (
        date(2023, 2, 10), None, "JPSS-2", "NOAA-21", 5200)


def test_es_inmutable():
    with pytest.raises(Exception):
        PRODUCTOS["VJ146A2"].fin = date(2030, 1, 1)


def test_el_por_omision_es_vj146a2():
    assert producto_por_omision().nombre == "VJ146A2"
    assert obtener_producto(None).nombre == "VJ146A2"


def test_ntl_producto_cambia_el_por_omision(monkeypatch):
    monkeypatch.setenv("NTL_PRODUCTO", "VNP46A2")
    assert obtener_producto(None).nombre == "VNP46A2"
    monkeypatch.setenv("NTL_PRODUCTO", "vj246a2")
    assert obtener_producto(None) is PRODUCTOS["VJ246A2"]


def test_ntl_producto_invalido_falla_con_mensaje_claro(monkeypatch):
    monkeypatch.setenv("NTL_PRODUCTO", "VNP46A1")
    with pytest.raises(ValueError, match="VNP46A2, VJ146A2"):
        obtener_producto(None)


def test_config_producto_es_el_nombre_por_omision():
    assert isinstance(config.PRODUCTO, str)


@pytest.mark.parametrize("malo", ["VNP46A1", "xyz", ""])
def test_producto_invalido(malo):
    with pytest.raises(ValueError, match="Soportados: VNP46A2, VJ146A2"):
        obtener_producto(malo)


def test_acepta_nombre_o_instancia():
    p = PRODUCTOS["VNP46A2"]
    assert obtener_producto(p) is p
    assert obtener_producto("vj146a2") is PRODUCTOS["VJ146A2"]


def test_url_listado():
    assert PRODUCTOS["VJ146A2"].url_listado(2026, 5) == (
        "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData/"
        "5200/VJ146A2/2026/005/")
    assert PRODUCTOS["VJ246A2"].url_listado(2026, 312) == (
        "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData/"
        "5200/VJ246A2/2026/312/")


def test_vigencia():
    vnp, vj = PRODUCTOS["VNP46A2"], PRODUCTOS["VJ146A2"]
    assert vnp.motivo_fuera_de_rango(date(2026, 11, 2)) is None
    assert "después de 2026-11-02; usar VJ146A2 o VJ246A2" in vnp.motivo_fuera_de_rango(date(2026, 11, 3))
    assert "antes de 2012-01-19" in vnp.motivo_fuera_de_rango(date(2012, 1, 18))
    assert vj.motivo_fuera_de_rango(date(2018, 1, 4)) is not None
    assert vj.motivo_fuera_de_rango(date(2018, 1, 5)) is None
    assert vj.motivo_fuera_de_rango(date(2040, 1, 1)) is None


def test_vigencia_vj246a2():
    vj2 = PRODUCTOS["VJ246A2"]
    assert "antes de 2023-02-10" in vj2.motivo_fuera_de_rango(date(2023, 2, 9))
    assert vj2.motivo_fuera_de_rango(date(2023, 2, 10)) is None
    assert vj2.motivo_fuera_de_rango(date(2040, 1, 1)) is None
