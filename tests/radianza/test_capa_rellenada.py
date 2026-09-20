"""
La capa rellenada: un valor todos los días, y la etiqueta que dice de cuándo es.

El gránulo VNP46A2 trae, junto a la medición del algoritmo principal,
`Gap_Filled_DNB_BRDF-Corrected_NTL` —que arrastra la última recuperación buena
donde no hubo ninguna— y `Latest_High_Quality_Retrieval`, la antigüedad de ese
valor en días.

Antes el pipeline devolvía None en cuanto el algoritmo principal no recuperaba
nada, y esa noche desaparecía de la serie. Medido sobre Cuauhtémoc en una
ventana de tres semanas: 13 de 16 fechas publicadas no tenían recuperación y
todas traían valor rellenado. Se estaba tirando lo que el satélite sí entregó.

Lo que estas pruebas fijan es el trato: se publica la capa, y nunca sin su
antigüedad. Un tramo plano de seis días es legítimo mientras diga que es un
tramo plano; sin la etiqueta se lee como "la luz no cambió".
"""
import h5py
import numpy as np
import pytest

from ntl.radianza.extraccion import process_image

GRUPO = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
FECHA = __import__("datetime").date(2026, 9, 1)
PESOS = [(x, y, 1.0) for x in range(4) for y in range(4)]


def _granulo(tmp_path, *, calidad, rellenada=None, antiguedad=None, principal=None):
    """Gránulo 10x10 con las capas que se le pidan."""
    ruta = tmp_path / "g.h5"
    with h5py.File(ruta, "w") as f:
        g = f.create_group(GRUPO)

        principal = np.full((10, 10), 50.0, dtype=np.float32) if principal is None else principal
        ds = g.create_dataset("DNB_BRDF-Corrected_NTL", data=principal)
        ds.attrs["_FillValue"] = np.array([-999.9], dtype=np.float32)
        ds.attrs["scale_factor"] = np.array([1.0])
        ds.attrs["add_offset"] = np.array([0.0])
        ds.attrs["units"] = np.bytes_(b"nWatts/(cm^2 sr)")

        g.create_dataset(
            "Mandatory_Quality_Flag",
            data=np.full((10, 10), calidad, dtype=np.uint8),
        )

        if rellenada is not None:
            dr = g.create_dataset("Gap_Filled_DNB_BRDF-Corrected_NTL", data=rellenada)
            dr.attrs["_FillValue"] = np.array([-999.9], dtype=np.float32)
            dr.attrs["scale_factor"] = np.array([1.0])
            dr.attrs["add_offset"] = np.array([0.0])
            dr.attrs["units"] = np.bytes_(b"nWatts/(cm^2 sr)")

        if antiguedad is not None:
            da = g.create_dataset("Latest_High_Quality_Retrieval", data=antiguedad)
            da.attrs["_FillValue"] = np.array([255], dtype=np.uint8)
            da.attrs["scale_factor"] = np.array([1.0])
            da.attrs["add_offset"] = np.array([0.0])
    return str(ruta)


def _procesar(ruta):
    return process_image(ruta, PESOS, FECHA, "prueba", delete_file=False)


# --- el caso que antes se perdía ---


def test_una_noche_sin_recuperacion_ya_produce_registro(tmp_path):
    """255 en toda la bandera: el algoritmo principal no recuperó nada.

    Antes esto devolvía None y la fecha desaparecía de la serie, aunque el
    gránulo traía la capa rellenada entera.
    """
    ruta = _granulo(
        tmp_path,
        calidad=255,
        rellenada=np.full((10, 10), 80.0, dtype=np.float32),
        antiguedad=np.full((10, 10), 5, dtype=np.uint8),
    )
    r = _procesar(ruta)

    assert r is not None, "la fecha no debe desaparecer de la serie"
    assert r.Radianza_rellenada is not None
    assert r.Radianza_rellenada.Media_de_radianza == pytest.approx(80.0)
    assert r.Radianza_rellenada.Fraccion_valida == pytest.approx(1.0)


def test_ese_registro_no_finge_haber_medido(tmp_path):
    """La capa principal sale en cero-área, no copiada del relleno.

    Si el relleno se copiara ahí, el registro diría "se midió 80" una noche en
    la que no se midió nada, y no habría forma de saberlo después.
    """
    ruta = _granulo(
        tmp_path,
        calidad=255,
        rellenada=np.full((10, 10), 80.0, dtype=np.float32),
        antiguedad=np.full((10, 10), 5, dtype=np.uint8),
    )
    r = _procesar(ruta)

    assert r.Fraccion_valida == 0.0
    assert r.Cantidad_de_pixeles == 0.0
    assert r.Media_de_radianza == 0.0
    assert r.Fraccion_medida == 0.0
    assert r.Antiguedad_mediana_dias == 5.0


def test_sin_ninguna_capa_sigue_sin_haber_registro(tmp_path):
    """No hay dato es no hay dato: eso sí debe seguir devolviendo None."""
    ruta = _granulo(tmp_path, calidad=255)
    assert _procesar(ruta) is None


# --- la antigüedad, que es lo que hace publicable la capa ---


def test_antiguedad_cero_significa_medido_esa_noche(tmp_path):
    ruta = _granulo(
        tmp_path,
        calidad=0,
        rellenada=np.full((10, 10), 50.0, dtype=np.float32),
        antiguedad=np.zeros((10, 10), dtype=np.uint8),
    )
    r = _procesar(ruta)
    assert r.Antiguedad_mediana_dias == 0.0
    assert r.Fraccion_medida == pytest.approx(1.0)


def test_la_fraccion_medida_separa_lo_medido_de_lo_arrastrado(tmp_path):
    """Media ciudad medida esa noche, media arrastrada de hace tres días."""
    antiguedad = np.zeros((10, 10), dtype=np.uint8)
    antiguedad[:, 2:] = 3  # las columnas x>=2 vienen arrastradas
    ruta = _granulo(
        tmp_path,
        calidad=0,
        rellenada=np.full((10, 10), 50.0, dtype=np.float32),
        antiguedad=antiguedad,
    )
    r = _procesar(ruta)
    # PESOS cubre x en 0..3: dos columnas medidas, dos arrastradas.
    assert r.Fraccion_medida == pytest.approx(0.5)


def test_donde_el_principal_recupera_las_dos_capas_coinciden(tmp_path):
    """Lo medido sobre gránulos reales: gap == main donde hay recuperación.

    Si dejaran de coincidir, la capa rellenada no sería un superconjunto de la
    medición y publicar las dos como comparables sería un error.
    """
    valores = np.random.uniform(10, 200, size=(10, 10)).astype(np.float32)
    ruta = _granulo(
        tmp_path,
        calidad=0,
        principal=valores,
        rellenada=valores.copy(),
        antiguedad=np.zeros((10, 10), dtype=np.uint8),
    )
    r = _procesar(ruta)
    assert r.Radianza_rellenada.Media_de_radianza == pytest.approx(r.Media_de_radianza)
    assert r.Radianza_rellenada.Suma_de_radianza == pytest.approx(r.Suma_de_radianza)


# --- compatibilidad ---


def test_un_granulo_sin_la_capa_sigue_procesandose(tmp_path):
    """Los gránulos antiguos no la traen; eso no debe romper nada."""
    ruta = _granulo(tmp_path, calidad=0)
    r = _procesar(ruta)
    assert r is not None
    assert r.Media_de_radianza == pytest.approx(50.0)
    assert r.Radianza_rellenada is None
    assert r.Antiguedad_mediana_dias is None


def test_la_capa_rellenada_ignora_la_bandera_de_calidad(tmp_path):
    """Filtrarla por la bandera la dejaría vacía justo cuando hace falta.

    Donde la bandera dice 255 es exactamente donde esta capa tiene algo que
    aportar; su propio _FillValue es el criterio de validez.
    """
    from ntl.core.lectura import leer_rellenada

    ruta = _granulo(
        tmp_path,
        calidad=255,
        rellenada=np.full((10, 10), 80.0, dtype=np.float32),
        antiguedad=np.full((10, 10), 2, dtype=np.uint8),
    )
    with h5py.File(ruta, "r") as f:
        matriz, meta = leer_rellenada(f)
    assert np.isfinite(matriz).all(), "la bandera no debe vaciar esta capa"
    assert meta["fraccion_valida"] == pytest.approx(1.0)


def test_la_matriz_rellenada_tiene_la_forma_del_recorte(tmp_path):
    ruta = _granulo(
        tmp_path,
        calidad=0,
        rellenada=np.full((10, 10), 80.0, dtype=np.float32),
        antiguedad=np.zeros((10, 10), dtype=np.uint8),
    )
    r = _procesar(ruta)
    assert len(r.Matriz_rellenada) == r.Filas
    assert len(r.Matriz_rellenada[0]) == r.Columnas
