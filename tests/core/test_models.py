"""Unit tests for ntl Pydantic models."""
import pytest
from datetime import date
from pydantic import ValidationError

from ntl.core.models import CoordenadasPixeles, MedicionResultado


class TestCoordenadasPixeles:
    def test_valid_pesos(self):
        obj = CoordenadasPixeles(
            cuadrante="h08v07", pesos=[(0, 0, 1.0), (1, 1, 0.5), (2, 2, 0.25)]
        )
        assert obj.cuadrante == "h08v07"
        assert obj.coordenadas_pixeles == [(0, 0), (1, 1), (2, 2)]
        assert obj.area == pytest.approx(1.75)

    def test_missing_cuadrante_raises(self):
        with pytest.raises(ValidationError):
            CoordenadasPixeles(pesos=[(0, 0, 1.0)])

    def test_missing_pesos_raises(self):
        with pytest.raises(ValidationError):
            CoordenadasPixeles(cuadrante="h08v07")

    def test_empty_pesos_allowed(self):
        obj = CoordenadasPixeles(cuadrante="h09v07", pesos=[])
        assert obj.coordenadas_pixeles == []
        assert obj.area == 0

    def test_formato_anterior_sigue_cargando(self):
        """Una tabla sin regenerar no debe romper el procesamiento.

        Se asume cobertura 1.0, que reproduce el sesgo de frontera anterior.
        """
        obj = CoordenadasPixeles(
            cuadrante="h08v07", coordenadas_pixeles=[(0, 0), (1, 1)]
        )
        assert obj.pesos == [(0, 0, 1.0), (1, 1, 1.0)]
        assert obj.area == 2.0

    def test_area_no_es_un_conteo(self):
        """El área suma coberturas; contar píxeles sobreestima la frontera."""
        obj = CoordenadasPixeles(
            cuadrante="h08v07", pesos=[(0, 0, 1.0), (1, 0, 0.5), (2, 0, 0.5)]
        )
        assert len(obj.pesos) == 3
        assert obj.area == pytest.approx(2.0)


class TestMedicionResultado:
    def test_valid_medicion_resultado(self):
        data = {
            "Fecha": date(2024, 1, 1),
            "Producto": "VJ146A2",
            "Cantidad_de_pixeles": 100,
            "Suma_de_radianza": 500.0,
            "Media_de_radianza": 5.0,
            "Desviacion_estandar_de_radianza": 2.0,
            "Maximo_de_radianza": 10.0,
            "Minimo_de_radianza": 0.5,
            "Percentil_25_de_radianza": 3.0,
            "Percentil_50_de_radianza": 5.0,
            "Percentil_75_de_radianza": 7.0,
        }
        obj = MedicionResultado(**data)
        assert obj.Fecha == date(2024, 1, 1)
        assert obj.Cantidad_de_pixeles == 100
        assert obj.Media_de_radianza == 5.0
        assert obj.Maximo_de_radianza == 10.0

    def test_dict_export(self):
        data = {
            "Fecha": date(2024, 1, 15),
            "Producto": "VNP46A2",
            "Cantidad_de_pixeles": 50,
            "Suma_de_radianza": 250.0,
            "Media_de_radianza": 5.0,
            "Desviacion_estandar_de_radianza": 1.0,
            "Maximo_de_radianza": 8.0,
            "Minimo_de_radianza": 2.0,
            "Percentil_25_de_radianza": 4.0,
            "Percentil_50_de_radianza": 5.0,
            "Percentil_75_de_radianza": 6.0,
        }
        obj = MedicionResultado(**data)
        d = obj.model_dump()
        assert "Fecha" in d
        assert "Media_de_radianza" in d
        assert d["Cantidad_de_pixeles"] == 50

    def test_producto_es_obligatorio_y_solo_admite_los_soportados(self):
        base = dict(
            Fecha=date(2024, 1, 1), Cantidad_de_pixeles=1, Suma_de_radianza=1.0,
            Media_de_radianza=1.0, Desviacion_estandar_de_radianza=0.0,
            Maximo_de_radianza=1.0, Minimo_de_radianza=1.0,
            Percentil_25_de_radianza=1.0, Percentil_50_de_radianza=1.0,
            Percentil_75_de_radianza=1.0)
        with pytest.raises(ValidationError):
            MedicionResultado(**base)
        with pytest.raises(ValidationError):
            MedicionResultado(Producto="VNP46A1", **base)
        assert MedicionResultado(Producto="VJ146A2", **base).Producto == "VJ146A2"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            MedicionResultado(
                Fecha=date(2024, 1, 1),
                Cantidad_de_pixeles=10,
                # missing Suma_de_radianza and the rest
            )

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            MedicionResultado(
                Fecha="not-a-date",
                Cantidad_de_pixeles=10,
                Suma_de_radianza=1.0,
                Media_de_radianza=1.0,
                Desviacion_estandar_de_radianza=0.0,
                Maximo_de_radianza=1.0,
                Minimo_de_radianza=0.0,
                Percentil_25_de_radianza=0.0,
                Percentil_50_de_radianza=0.0,
                Percentil_75_de_radianza=0.0,
            )
