"""Integration tests for ntl processing with mocked HDF5."""
from datetime import date

import pytest

from ntl.core.errores import MedicionImposible
from ntl.radianza.extraccion import extract_radiance_matrix, process_image


class TestProcessImage:
    def test_returns_medicion_resultado_when_path_valid(self, sample_hdf5_path):
        # Coordinates within 10x10 fixture image
        coords = [(1, 1), (2, 1), (2, 2), (1, 2)]
        result = process_image(
            sample_hdf5_path,
            coords,
            date(2024, 1, 1),
            "Iztapalapa",
            delete_file=False,
        )
        assert result is not None
        assert result.Municipio == "Iztapalapa"
        assert result.Fecha == date(2024, 1, 1)
        assert result.Cantidad_de_pixeles == len(coords)
        assert result.Suma_de_radianza >= 0
        assert result.Media_de_radianza >= 0

    def test_returns_none_when_path_does_not_exist(self):
        result = process_image(
            "/nonexistent/path.h5",
            [(0, 0)],
            date(2024, 1, 1),
            "Test",
            delete_file=False,
        )
        assert result is None

    def test_lanza_si_ninguna_coordenada_cae_en_la_reticula(self, sample_hdf5_path):
        """
        La tabla de cobertura y la imagen no cuadran: eso no es «no hay
        medición», es una incoherencia, y devolverla como None la confundía con
        una noche nublada.
        """
        coords = [(100, 100), (101, 101)]  # fuera de la imagen de 10x10
        with pytest.raises(MedicionImposible, match="ninguna coordenada"):
            process_image(
                sample_hdf5_path,
                coords,
                date(2024, 1, 1),
                "Test",
                delete_file=False,
            )

    def test_lanza_si_la_cobertura_viene_vacia(self, sample_hdf5_path):
        with pytest.raises(MedicionImposible):
            process_image(
                sample_hdf5_path,
                [],
                date(2024, 1, 1),
                "Test",
                delete_file=False,
            )


class TestExtractRadianceMatrix:
    def test_returns_dict_with_matrices_when_valid(self, sample_hdf5_path):
        coords = [(1, 1), (2, 1), (2, 2), (1, 2)]
        result = extract_radiance_matrix(
            sample_hdf5_path,
            coords,
            date(2024, 1, 1),
            "iztapalapa",
        )
        assert result is not None
        assert result["municipio"] == "iztapalapa"
        assert result["fecha"] == date(2024, 1, 1)
        assert result["bbox"] == {"min_x": 1, "max_x": 2, "min_y": 1, "max_y": 2}
        assert result["rows"] == 2
        assert result["cols"] == 2
        assert len(result["radiance_matrix"]) == 2
        assert len(result["radiance_matrix"][0]) == 2
        assert len(result["municipality_mask"]) == 2
        assert len(result["municipality_mask"][0]) == 2
        assert sum(sum(row) for row in result["municipality_mask"]) == 4

    def test_returns_none_when_path_does_not_exist(self):
        result = extract_radiance_matrix(
            "/nonexistent/path.h5",
            [(0, 0)],
            date(2024, 1, 1),
            "Test",
        )
        assert result is None

    def test_lanza_si_ninguna_coordenada_cae_en_la_reticula(self, sample_hdf5_path):
        coords = [(100, 100), (101, 101)]
        with pytest.raises(MedicionImposible):
            extract_radiance_matrix(
                sample_hdf5_path,
                coords,
                date(2024, 1, 1),
                "Test",
            )


class TestPonderacionPorCobertura:
    """La cobertura fraccional debe llegar hasta las métricas."""

    def test_la_cobertura_reduce_el_area_y_la_suma(self, sample_hdf5_path):
        completos = [(1, 1, 1.0), (2, 1, 1.0), (2, 2, 1.0), (1, 2, 1.0)]
        frontera = [(1, 1, 1.0), (2, 1, 0.5), (2, 2, 1.0), (1, 2, 0.5)]

        entero = process_image(sample_hdf5_path, completos, date(2024, 1, 1),
                               "Iztapalapa", delete_file=False)
        medio = process_image(sample_hdf5_path, frontera, date(2024, 1, 1),
                              "Iztapalapa", delete_file=False)

        assert entero.Cantidad_de_pixeles == pytest.approx(4.0)
        assert medio.Cantidad_de_pixeles == pytest.approx(3.0)
        assert medio.Suma_de_radianza < entero.Suma_de_radianza

    def test_area_no_es_el_numero_de_pixeles(self, sample_hdf5_path):
        pesos = [(1, 1, 0.25), (2, 1, 0.25)]
        r = process_image(sample_hdf5_path, pesos, date(2024, 1, 1),
                          "Iztapalapa", delete_file=False)
        assert len(pesos) == 2
        assert r.Cantidad_de_pixeles == pytest.approx(0.5)

    def test_expone_la_matriz_de_cobertura(self, sample_hdf5_path):
        pesos = [(1, 1, 1.0), (2, 1, 0.5), (2, 2, 1.0), (1, 2, 0.25)]
        r = process_image(sample_hdf5_path, pesos, date(2024, 1, 1),
                          "Iztapalapa", delete_file=False)
        assert r.Cobertura_municipio == [[1.0, 0.5], [0.25, 1.0]]
        # La máscara binaria sigue marcando qué píxeles toca el municipio
        assert r.Mascara_municipio == [[1, 1], [1, 1]]

    def test_el_formato_anterior_sigue_funcionando(self, sample_hdf5_path):
        """Una tabla sin regenerar asume cobertura 1.0 y avisa."""
        coords = [(1, 1), (2, 1), (2, 2), (1, 2)]
        r = process_image(sample_hdf5_path, coords, date(2024, 1, 1),
                          "Iztapalapa", delete_file=False)
        assert r is not None
        assert r.Cantidad_de_pixeles == pytest.approx(4.0)


@pytest.fixture
def hdf5_georreferenciado(tmp_path):
    """
    HDF5 minimo cuyo StructMetadata.0 declara ser el cuadrante h08v07.

    `sample_hdf5_path` trae metadatos inventados, que servian mientras nadie
    los leia. Ahora `process_image` los comprueba, asi que hace falta un archivo
    que este donde dice estar. Las cifras son las del granulo real
    VNP46A2.A2024015.h08v07.002: grados por un millon, no metros.
    """
    import h5py
    import numpy as np

    ruta = tmp_path / "h08v07.h5"
    metadata = (
        "XDim=2400\n"
        "YDim=2400\n"
        "UpperLeftPointMtrs=(-100000000.000000,20000000.000000)\n"
        "LowerRightMtrs=(-90000000.000000,10000000.000000)\n"
        "Projection=HE5_GCTP_GEO\n"
        "GridOrigin=HE5_HDFE_GD_UL\n"
    )
    with h5py.File(ruta, "w") as f:
        grupo = f.create_group("HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields")
        ds = grupo.create_dataset(
            "DNB_BRDF-Corrected_NTL",
            data=np.arange(2400 * 2400, dtype=np.float32).reshape(2400, 2400) % 100,
        )
        ds.attrs["_FillValue"] = np.float32(-999.9)
        ds.attrs["scale_factor"] = 1.0
        ds.attrs["units"] = b"nWatts/(cm^2 sr)"
        grupo.create_dataset(
            "Mandatory_Quality_Flag", data=np.zeros((2400, 2400), dtype=np.uint8)
        )
        info = f.create_group("HDFEOS INFORMATION")
        info.create_dataset(
            "StructMetadata.0",
            data=np.array(metadata, dtype="S" + str(len(metadata) + 1)),
        )
    return str(ruta)


class TestVerificacionDeGeorreferencia:
    """
    Antes de tocar la radianza, el archivo tiene que estar donde se supone.

    Las coberturas se precalculan contra un origen derivado del identificador
    del cuadrante. Si el producto cambiara de convencion, todos los municipios
    se desplazarian en silencio y ninguna prueba de area lo notaria, porque
    todas validan contra ese mismo supuesto. Un pixel de desalineamiento mueve
    la media de radianza municipal un 4.6% en la mediana de las alcaldias
    (ver scripts/sensibilidad_desplazamiento.py).
    """

    PESOS = [(1200, 1200, 1.0), (1201, 1200, 0.5)]

    def test_procesa_cuando_la_georreferencia_coincide(self, hdf5_georreferenciado):
        datos = process_image(
            hdf5_georreferenciado, self.PESOS, date(2024, 1, 15), "prueba",
            delete_file=False, cuadrante="h08v07",
        )
        assert datos is not None
        assert datos.Cantidad_de_pixeles == pytest.approx(1.5)

    def test_no_devuelve_datos_si_el_cuadrante_no_coincide(self, hdf5_georreferenciado, capsys):
        datos = process_image(
            hdf5_georreferenciado, self.PESOS, date(2024, 1, 15), "prueba",
            delete_file=False, cuadrante="h09v07",
        )
        assert datos is None
        assert "Georreferencia incoherente" in capsys.readouterr().out

    def test_sin_cuadrante_no_se_comprueba_nada(self, hdf5_georreferenciado):
        """La verificacion es opcional: el resto del proyecto llama sin ella."""
        datos = process_image(
            hdf5_georreferenciado, self.PESOS, date(2024, 1, 15), "prueba",
            delete_file=False,
        )
        assert datos is not None

    def test_extract_radiance_matrix_tambien_comprueba(self, hdf5_georreferenciado):
        assert extract_radiance_matrix(
            hdf5_georreferenciado, self.PESOS, date(2024, 1, 15), "prueba",
            cuadrante="h08v07",
        ) is not None
        assert extract_radiance_matrix(
            hdf5_georreferenciado, self.PESOS, date(2024, 1, 15), "prueba",
            cuadrante="h09v07",
        ) is None
