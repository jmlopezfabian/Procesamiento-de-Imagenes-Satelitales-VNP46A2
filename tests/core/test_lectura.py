"""
Lectura de la radianza: unidades físicas y valores de relleno.

El histórico trae cuatro fechas en las que el cuadrante completo venía relleno
y 65535 se sumó como si fuera radianza; las sumas salieron 229 veces mayores
que las normales. Estas pruebas cubren esa clase de fallo.
"""
import h5py
import numpy as np
import pytest

from ntl.core.lectura import leer_radianza
from ntl.core.metricas import metricas_ponderadas

GRUPO = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
RUTA = f"{GRUPO}/DNB_BRDF-Corrected_NTL"
RUTA_CALIDAD = f"{GRUPO}/Mandatory_Quality_Flag"


def _hdf5(tmp_path, datos, calidad=None, tipo=np.uint16, producto=b"VNP46A2", **attrs):
    """
    Archivo con la forma de VNP46A2.

    La bandera de calidad es obligatoria, así que por omisión se llena con 0
    —alta calidad— para aislar lo que cada prueba quiere comprobar.
    """
    datos = np.asarray(datos)
    ruta = tmp_path / "prueba.h5"
    with h5py.File(ruta, "w") as f:
        if producto is not None:
            f.attrs["ShortName"] = producto
        ds = f.create_dataset(RUTA, data=datos.astype(tipo))
        for k, v in attrs.items():
            ds.attrs[k] = v
        if calidad is None:
            calidad = np.zeros(datos.shape, dtype=np.uint8)
        f.create_dataset(RUTA_CALIDAD, data=np.asarray(calidad, dtype=np.uint8))
    return ruta


class TestUnidadesFisicas:
    def test_aplica_el_factor_de_escala(self, tmp_path):
        ruta = _hdf5(tmp_path, np.array([[100, 200]]), scale_factor=0.1,
                     add_offset=0.0, units=b"nW/(cm2 sr)")
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert radianza.tolist() == [[10.0, 20.0]]
        assert meta["unidades"] == "nW/(cm2 sr)"

    def test_aplica_el_desplazamiento(self, tmp_path):
        ruta = _hdf5(tmp_path, np.array([[10]]), scale_factor=2.0, add_offset=5.0)
        with h5py.File(ruta) as f:
            radianza, _ = leer_radianza(f)
        assert radianza[0, 0] == pytest.approx(25.0)

    def test_sin_atributos_no_transforma(self, tmp_path):
        ruta = _hdf5(tmp_path, np.array([[7, 9]]))
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert radianza.tolist() == [[7.0, 9.0]]
        assert meta["escala"] == 1.0


class TestValoresDeRelleno:
    def test_el_relleno_sale_como_nan(self, tmp_path):
        ruta = _hdf5(tmp_path, np.array([[100, 65535, 300]]),
                     scale_factor=0.1, _FillValue=65535)
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert np.isnan(radianza[0, 1])
        assert radianza[0, 0] == pytest.approx(10.0)
        assert meta["invalidos"] == 1

    def test_respeta_el_rango_valido(self, tmp_path):
        ruta = _hdf5(tmp_path, np.array([[5, 50, 500]]), scale_factor=1.0,
                     valid_min=10, valid_max=100)
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert np.isnan(radianza[0, 0]) and np.isnan(radianza[0, 2])
        assert radianza[0, 1] == pytest.approx(50.0)
        assert meta["invalidos"] == 2

    def test_el_relleno_no_se_suma_como_radianza(self, tmp_path):
        """El fallo exacto de 2024-06-17, cuando el cuadrante entero era relleno."""
        ruta = _hdf5(tmp_path, np.array([[65535, 65535]]),
                     scale_factor=0.1, _FillValue=65535)
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert meta["fraccion_valida"] == 0.0
        # Sin la máscara esto habría dado 13107.0 en vez de None
        assert metricas_ponderadas(radianza.ravel(), np.ones(2)) is None


class TestFraccionValida:
    def test_la_perdida_de_territorio_queda_registrada(self):
        valores = np.array([100.0, np.nan, 100.0, 100.0])
        m = metricas_ponderadas(valores, np.ones(4))
        assert m["Fraccion_valida"] == pytest.approx(0.75)
        assert m["Cantidad_de_pixeles"] == pytest.approx(3.0)

    def test_sin_perdida_la_fraccion_es_uno(self):
        m = metricas_ponderadas(np.array([1.0, 2.0]), np.ones(2))
        assert m["Fraccion_valida"] == pytest.approx(1.0)

    def test_la_suma_no_incluye_los_pixeles_sin_medicion(self):
        con_hueco = metricas_ponderadas(np.array([10.0, np.nan]), np.ones(2))
        assert con_hueco["Suma_de_radianza"] == pytest.approx(10.0)

    def test_el_area_encoge_con_los_pixeles_perdidos(self):
        """Es lo honesto: el territorio sin dato no se puede reportar como medido."""
        m = metricas_ponderadas(np.array([10.0, np.nan]), np.array([1.0, 0.5]))
        assert m["Cantidad_de_pixeles"] == pytest.approx(1.0)
        assert m["Fraccion_valida"] == pytest.approx(1.0 / 1.5)


def _hdf5_a2(tmp_path, ntl, calidad=None):
    """Radianza en punto flotante, como la publica VNP46A2."""
    return _hdf5(tmp_path, np.asarray(ntl, dtype=np.float32), calidad=calidad,
                 tipo=np.float32, _FillValue=np.float32(-999.9),
                 scale_factor=1.0, units=b"nWatts/(cm^2 sr)")


class TestVNP46A2:
    def test_descarta_los_pixeles_de_mala_calidad(self, tmp_path):
        """0 es alta calidad; de 1 a 5 son distintas formas de mala."""
        ruta = _hdf5_a2(tmp_path, [[10.0, 20.0, 30.0, 40.0]], calidad=[[0, 1, 5, 0]])
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert radianza[0, 0] == pytest.approx(10.0)
        assert np.isnan(radianza[0, 1]) and np.isnan(radianza[0, 2])
        assert radianza[0, 3] == pytest.approx(40.0)
        assert meta["producto"] == "VNP46A2"

    def test_el_relleno_en_punto_flotante_se_reconoce(self, tmp_path):
        """El relleno de VNP46A2 es -999.9; comparar por igualdad exacta es frágil."""
        ruta = _hdf5_a2(tmp_path, [[-999.9, 12.5]], calidad=[[0, 0]])
        with h5py.File(ruta) as f:
            radianza, _ = leer_radianza(f)
        assert np.isnan(radianza[0, 0])
        assert radianza[0, 1] == pytest.approx(12.5)

    def test_una_noche_nublada_no_deja_nada_medible(self, tmp_path):
        """El caso del 2025-01-05: VNP46A1 dio un número, VNP46A2 dice que no hay dato."""
        ruta = _hdf5_a2(tmp_path, [[10.0, 20.0]], calidad=[[255, 255]])
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert meta["fraccion_valida"] == 0.0
        assert metricas_ponderadas(radianza.ravel(), np.ones(2)) is None

    def test_un_archivo_sin_bandera_se_rechaza(self, tmp_path):
        """Sin Mandatory_Quality_Flag no hay forma de saber si la noche servía.

        VNP46A1 no la trae. Procesarlo sin filtrar produciría una serie que
        parece válida y no lo es, así que se rechaza en vez de continuar.
        """
        ruta = tmp_path / "sin_bandera.h5"
        with h5py.File(ruta, "w") as f:
            f.attrs["ShortName"] = b"VNP46A2"
            f.create_dataset(RUTA, data=np.array([[10.0, 20.0]], dtype=np.float32))
        with h5py.File(ruta) as f:
            with pytest.raises(ValueError, match="Mandatory_Quality_Flag"):
                leer_radianza(f)


class TestProductoDelArchivo:
    """El producto sale del atributo raíz ShortName, no de la configuración."""

    @pytest.mark.parametrize("corto", [b"VNP46A2", b"VJ146A2", "VJ146A2"])
    def test_lo_toma_del_archivo(self, tmp_path, corto):
        ruta = _hdf5(tmp_path, [[10, 20]], producto=corto)
        with h5py.File(ruta) as f:
            _, meta = leer_radianza(f)
        assert meta["producto"] == (corto.decode() if isinstance(corto, bytes) else corto)

    def test_vnp46a1_se_rechaza(self, tmp_path):
        ruta = _hdf5(tmp_path, [[10, 20]], producto=b"VNP46A1")
        with h5py.File(ruta) as f:
            with pytest.raises(ValueError, match="VNP46A1.*VNP46A2, VJ146A2"):
                leer_radianza(f)

    def test_sin_shortname_se_rechaza(self, tmp_path):
        ruta = _hdf5(tmp_path, [[10, 20]], producto=None)
        with h5py.File(ruta) as f:
            with pytest.raises(ValueError, match="ausente"):
                leer_radianza(f)
