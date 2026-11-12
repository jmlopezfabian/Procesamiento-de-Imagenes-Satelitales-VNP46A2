"""El entorno de quien corre las pruebas no debe cambiar su resultado."""
import os

from ntl.core.productos import producto_por_omision


def test_ntl_producto_exportada_no_afecta_a_las_pruebas():
    # conftest la quita en cada prueba (autouse): con NTL_PRODUCTO=VNP46A2
    # exportada fallaban 15 pruebas que esperan el VJ146A2 por omisión.
    assert "NTL_PRODUCTO" not in os.environ
    assert producto_por_omision().nombre == "VJ146A2"
