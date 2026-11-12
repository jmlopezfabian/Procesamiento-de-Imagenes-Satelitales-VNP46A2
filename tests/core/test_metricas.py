"""Métricas ponderadas por cobertura: la única implementación del proyecto."""
import numpy as np
import pytest

from ntl.core.metricas import metricas_ponderadas


class TestMetricasPonderadas:
    def test_pesos_uniformes_equivalen_al_promedio_simple(self):
        valores = np.array([10.0, 20.0, 30.0, 40.0])
        m = metricas_ponderadas(valores, np.ones(4))
        assert m["Cantidad_de_pixeles"] == pytest.approx(4.0)
        assert m["Suma_de_radianza"] == pytest.approx(100.0)
        assert m["Media_de_radianza"] == pytest.approx(25.0)
        assert m["Desviacion_estandar_de_radianza"] == pytest.approx(np.std(valores))

    def test_la_cobertura_pondera_la_suma(self):
        valores = np.array([100.0, 100.0])
        completo = metricas_ponderadas(valores, np.array([1.0, 1.0]))
        frontera = metricas_ponderadas(valores, np.array([1.0, 0.5]))
        assert completo["Suma_de_radianza"] == pytest.approx(200.0)
        assert frontera["Suma_de_radianza"] == pytest.approx(150.0)
        # La media no cambia: los dos píxeles valen lo mismo
        assert frontera["Media_de_radianza"] == pytest.approx(100.0)

    def test_area_es_la_suma_de_coberturas_no_el_conteo(self):
        m = metricas_ponderadas(np.array([1.0, 1.0, 1.0]), np.array([1.0, 0.5, 0.25]))
        assert m["Cantidad_de_pixeles"] == pytest.approx(1.75)

    def test_un_pixel_a_medias_desplaza_la_media(self):
        """El caso que motivó el cambio: la frontera no vale lo mismo."""
        valores = np.array([1000.0, 0.0])  # centro brillante, periferia oscura
        sin_ponderar = metricas_ponderadas(valores, np.array([1.0, 1.0]))
        ponderado = metricas_ponderadas(valores, np.array([1.0, 0.2]))
        assert sin_ponderar["Media_de_radianza"] == pytest.approx(500.0)
        assert ponderado["Media_de_radianza"] == pytest.approx(1000 / 1.2)

    def test_percentiles_sobre_la_masa_de_area(self):
        # El valor bajo casi no pesa, así que la mediana se va hacia el alto
        valores = np.array([0.0, 100.0])
        m = metricas_ponderadas(valores, np.array([0.01, 1.0]))
        assert m["Percentil_50_de_radianza"] > 50

    def test_maximo_y_minimo_ignoran_la_cobertura(self):
        m = metricas_ponderadas(np.array([5.0, 50.0]), np.array([0.1, 0.1]))
        assert m["Maximo_de_radianza"] == 50.0
        assert m["Minimo_de_radianza"] == 5.0

    def test_sin_pixeles_devuelve_none(self):
        assert metricas_ponderadas(np.array([]), np.array([])) is None

    def test_cobertura_total_nula_devuelve_none(self):
        assert metricas_ponderadas(np.array([1.0, 2.0]), np.array([0.0, 0.0])) is None

    def test_devuelve_los_campos_de_medicion_resultado(self):
        from ntl.core.models import MedicionResultado
        from datetime import date

        m = metricas_ponderadas(np.array([1.0, 2.0]), np.array([1.0, 0.5]))
        # No debe faltar ni sobrar ningún campo requerido
        MedicionResultado(Fecha=date(2024, 1, 1), Producto="VJ146A2", **m)
