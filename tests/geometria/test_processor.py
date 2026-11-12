"""Integration tests for ntl processor with mocked download and I/O."""
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from ntl.geometria.processor import SatelliteProcessor


def _fake_cobertura(poligono):
    """Cobertura de un recorte pequeño: interior lleno y un borde a medias."""
    pesos = np.array([
        [0.0, 0.5, 0.5, 0.0],
        [0.5, 1.0, 1.0, 0.5],
        [0.0, 0.5, 0.5, 0.0],
    ])
    return pesos, 2, 3      # matriz, fila y columna de origen dentro del cuadrante


class TestSatelliteProcessor:
    def test_get_measures_returns_dict_when_mocks_ok(self, sample_hdf5_path):
        geometria = Polygon([(-1.05, -0.60), (-1.04, -0.59), (-1.045, -0.58)])
        with patch("ntl.geometria.processor.find_file", return_value="http://example.com/file.h5"):
            with patch("ntl.geometria.processor.download_file", return_value=sample_hdf5_path):
                with patch("ntl.geometria.processor.extraer_geometria", return_value=geometria):
                    with patch("ntl.geometria.processor.cobertura_exacta", side_effect=_fake_cobertura):
                        proc = SatelliteProcessor("Iztapalapa")
                        result = proc.get_measures("01-01-24", "h08v07", show_plots=False)
        assert result is not None
        assert isinstance(result, dict)
        assert "Fecha" in result
        assert "Cantidad_de_pixeles" in result
        assert "Media_de_radianza" in result
        assert "Suma_de_radianza" in result

    def test_get_measures_returns_none_when_find_file_fails(self):
        with patch("ntl.geometria.processor.find_file", return_value=None):
            proc = SatelliteProcessor("Iztapalapa")
            result = proc.get_measures("01-01-24", "h08v07", show_plots=False)
        assert result is None

    def test_get_measures_returns_none_when_download_fails(self):
        with patch("ntl.geometria.processor.find_file", return_value="http://example.com/file.h5"):
            with patch("ntl.geometria.processor.download_file", return_value=None):
                proc = SatelliteProcessor("Iztapalapa")
                result = proc.get_measures("01-01-24", "h08v07", show_plots=False)
        assert result is None

    def test_get_measures_returns_none_when_extraer_geometria_fails(self, sample_hdf5_path):
        with patch("ntl.geometria.processor.find_file", return_value="http://example.com/file.h5"):
            with patch("ntl.geometria.processor.download_file", return_value=sample_hdf5_path):
                with patch("ntl.geometria.processor.extraer_geometria", return_value=None):
                    proc = SatelliteProcessor("Iztapalapa")
                    result = proc.get_measures("01-01-24", "h08v07", show_plots=False)
        assert result is None

    def test_run_accumulates_results_in_dataframe(self, sample_hdf5_path):
        geometria = Polygon([(-1.05, -0.60), (-1.04, -0.59), (-1.045, -0.58)])
        with patch("ntl.geometria.processor.find_file", return_value="http://example.com/file.h5"):
            with patch("ntl.geometria.processor.download_file", return_value=sample_hdf5_path):
                with patch("ntl.geometria.processor.extraer_geometria", return_value=geometria):
                    with patch("ntl.geometria.processor.cobertura_exacta", side_effect=_fake_cobertura):
                        with patch("ntl.geometria.processor.os.remove"):
                            proc = SatelliteProcessor("Iztapalapa")
                            df = proc.run(["01-01-24", "02-01-24"], "h08v07", show_plots=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Fecha" in df.columns
        assert "Media_de_radianza" in df.columns

    def test_run_returns_empty_dataframe_when_no_results(self):
        with patch("ntl.geometria.processor.find_file", return_value=None):
            proc = SatelliteProcessor("Iztapalapa")
            df = proc.run(["01-01-24", "02-01-24"], "h08v07", show_plots=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestMunicipioQueNoCabeEnElCuadrante:
    """
    La ruta síncrona procesa un cuadrante a la vez. Un municipio repartido no
    cabe, y recortar con índices fuera de rango no falla en numpy: devuelve
    píxeles del otro extremo de la imagen, que es un resultado plausible y
    equivocado. Tiene que avisar.
    """

    def test_avisa_en_vez_de_recortar_por_el_otro_lado(self):
        import numpy as np
        from shapely.geometry import Polygon
        from ntl.geometria.processor import _verificar_que_cabe

        # Polígono que se sale por la derecha del cuadrante.
        fuera = Polygon([(2390, 10), (2410, 10), (2410, 30), (2390, 30)])
        with pytest.raises(ValueError, match="no cabe en h08v07"):
            _verificar_que_cabe(fuera, (2400, 2400), "Sintetico", "h08v07")

    def test_no_molesta_a_un_municipio_que_si_cabe(self):
        from shapely.geometry import Polygon
        from ntl.geometria.processor import _verificar_que_cabe

        dentro = Polygon([(10, 10), (30, 10), (30, 30), (10, 30)])
        _verificar_que_cabe(dentro, (2400, 2400), "Sintetico", "h08v07")


class TestProductoEnElProcesador:
    def test_fuera_de_vigencia_no_pide_nada(self):
        with patch("ntl.geometria.processor.find_file") as find:
            proc = SatelliteProcessor("Iztapalapa", producto="VNP46A2")
            assert proc.get_measures("03-11-26", "h08v07", show_plots=False) is None
        find.assert_not_called()

    def test_el_producto_se_pasa_a_find_file(self):
        with patch("ntl.geometria.processor.find_file", return_value=None) as find:
            SatelliteProcessor("Iztapalapa", producto="VNP46A2").get_measures(
                "01-01-24", "h08v07", show_plots=False)
        assert find.call_args.args[3].nombre == "VNP46A2"

    def test_producto_invalido_falla_al_construir(self):
        with pytest.raises(ValueError, match="no soportado"):
            SatelliteProcessor("Iztapalapa", producto="VNP46A1")
