"""
Municipios que no caben en un cuadrante.

La retícula de Black Marble se corta cada 10 grados por conveniencia del
archivo, no por ninguna frontera administrativa: un municipio pegado a un
múltiplo de 10 en longitud cae en dos cuadrantes, en latitud cae en dos, y cerca
de una esquina de la retícula cae en cuatro. El pipeline los rechazaba.

Estas pruebas no comprueban que el código haga lo que hace, sino dos invariantes
que se pueden enunciar sin mirarlo:

1. **Partición.** El corte en cuadrantes no crea ni destruye territorio. La suma
   de las áreas de las piezas es el área que da el cálculo en una retícula sin
   cortar, y ningún píxel aparece en dos piezas.
2. **Indiferencia a la retícula.** La misma figura, medida en el centro de un
   cuadrante o a caballo entre cuatro, da el mismo resultado. Un municipio no
   puede cambiar de tamaño ni de radianza porque la NASA decidiera cortar el
   archivo por donde lo cortó.

Las escenas son sintéticas y con retícula reducida (120 px por cuadrante en vez
de 2400) para que cuatro imágenes quepan en memoria; el código deriva la
resolución de la forma del archivo, así que el caso es el mismo.
"""
from datetime import date

import h5py
import numpy as np
import pytest

from ntl.core.metricas import metricas_ponderadas
from ntl.core.models import CoordenadasPixeles
from ntl.core.utils import (
    cuadrante_de_coordenadas,
    cuadrantes_de_coordenadas,
    esquina_superior_izquierda,
)
from ntl.geometria.cobertura import cobertura_exacta
from ntl.geometria.mosaico import (
    a_marco_referencia,
    cobertura_por_cuadrante,
    cuadrante_referencia,
    poligono_en_pixeles_globales,
)
from ntl.core.errores import MedicionImposible
from ntl.radianza.extraccion import process_image_mosaico

# Cuadrante de prueba: 120 px de lado, 1/12 de grado por píxel.
FORMA = (120, 120)
GRADO_PX = FORMA[0] / 10.0

# Esquina de la retícula donde se juntan cuatro cuadrantes: lon -90, lat 10.
ESQUINA = (-90.0, 10.0)
CUATRO = ["h08v07", "h09v07", "h08v08", "h09v08"]


def cuadrado(centro, lado):
    """Cuadrado lon/lat cerrado, en coordenadas geográficas."""
    (cx, cy), h = centro, lado / 2
    return np.array([[cx - h, cy - h], [cx + h, cy - h],
                     [cx + h, cy + h], [cx - h, cy + h], [cx - h, cy - h]])


def ele(centro, lado):
    """
    Figura en L alrededor de un punto: su envolvente toca cuatro cuadrantes pero
    la geometría solo entra en tres. Es el caso que distingue "cuántos cuadrantes
    abarca la caja" de "cuántas imágenes hay que descargar".
    """
    (cx, cy), h = centro, lado / 2
    return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy],
                     [cx, cy], [cx, cy + h], [cx - h, cy + h], [cx - h, cy - h]])


def triangulo(centro, lado):
    """
    Triángulo cuya hipotenusa pasa por la esquina de la retícula: la envolvente
    toca cuatro cuadrantes y la figura solo tres, porque el cuadrante nordeste
    queda entero al otro lado de la hipotenusa.
    """
    (cx, cy), h = centro, lado / 2
    return np.array([[cx - h, cy - h], [cx + h, cy - h],
                     [cx - h, cy + h], [cx - h, cy - h]])


def banda_diagonal(centro, lado):
    """Banda estrecha por la diagonal: roza los otros dos cuadrantes en la esquina."""
    (cx, cy), h = centro, lado / 2
    d = lado / 12
    return np.array([[cx - h, cy - h + d], [cx - h + d, cy - h],
                     [cx + h, cy + h - d], [cx + h - d, cy + h],
                     [cx - h, cy - h + d]])


def radianza_sintetica(x_global, y_global):
    """
    Patrón determinista y distinto en cada píxel del planeta.

    Que dependa de la posición *global* es el punto: si el mosaico se compusiera
    con un desplazamiento equivocado, los valores no coincidirían con los del
    oráculo y la prueba lo vería, cosa que no pasaría con una imagen constante.
    """
    return (x_global * 7 + y_global * 13) % 97 + 0.5


def crear_cuadrante(directorio, cuadrante, relleno=False, calidad_mala=False,
                    forma=None):
    """
    HDF5 mínimo con la forma de VNP46A2, georreferenciado donde dice estar.

    `verificar_georreferencia` compara StructMetadata.0 con la esquina que se
    deriva del identificador, así que el archivo tiene que ser coherente o el
    procesamiento —con razón— se niega a usarlo.
    """
    h, v = int(cuadrante[1:3]), int(cuadrante[4:6])
    alto, ancho = forma or FORMA
    ys, xs = np.mgrid[0:alto, 0:ancho]
    matriz = radianza_sintetica(h * ancho + xs, v * alto + ys).astype(np.float32)
    if relleno:
        matriz[:] = -999.9

    ul_lon, ul_lat = esquina_superior_izquierda(cuadrante)
    metadata = (
        f"XDim={ancho}\nYDim={alto}\n"
        f"UpperLeftPointMtrs=({ul_lon * 1e6:.6f},{ul_lat * 1e6:.6f})\n"
        f"LowerRightMtrs=({(ul_lon + 10) * 1e6:.6f},{(ul_lat - 10) * 1e6:.6f})\n"
        "Projection=HE5_GCTP_GEO\nGridOrigin=HE5_HDFE_GD_UL\n"
    )

    ruta = directorio / f"{cuadrante}.h5"
    with h5py.File(ruta, "w") as f:
        grupo = f.create_group("HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields")
        ds = grupo.create_dataset("DNB_BRDF-Corrected_NTL", data=matriz)
        ds.attrs["_FillValue"] = np.float32(-999.9)
        ds.attrs["scale_factor"] = 1.0
        ds.attrs["units"] = b"nWatts/(cm^2 sr)"
        calidad = np.full((alto, ancho), 1 if calidad_mala else 0, dtype=np.uint8)
        grupo.create_dataset("Mandatory_Quality_Flag", data=calidad)
        info = f.create_group("HDFEOS INFORMATION")
        info.create_dataset(
            "StructMetadata.0",
            data=np.array(metadata, dtype="S" + str(len(metadata) + 1)),
        )
    return str(ruta)


def cobertura_global(coordenadas):
    """Cobertura en la retícula global, sin cortar por cuadrantes. El oráculo."""
    return cobertura_exacta(poligono_en_pixeles_globales(coordenadas, FORMA))


class TestCuantosCuadrantes:
    """Cuántas imágenes hace falta descargar, antes de descargar ninguna."""

    def test_una_alcaldia_cabe_en_uno(self):
        from ntl.core.utils import extraer_coordenadas
        assert cuadrantes_de_coordenadas(extraer_coordenadas("Iztapalapa")) == ["h08v07"]

    def test_dos_cuadrantes_en_horizontal(self):
        coords = cuadrado((-90.0, 15.0), 1.0)
        assert cuadrantes_de_coordenadas(coords) == ["h08v07", "h09v07"]

    def test_dos_cuadrantes_en_vertical(self):
        coords = cuadrado((-95.0, 10.0), 1.0)
        assert cuadrantes_de_coordenadas(coords) == ["h08v07", "h08v08"]

    def test_cuatro_cuadrantes_en_la_esquina(self):
        assert cuadrantes_de_coordenadas(cuadrado(ESQUINA, 1.0)) == CUATRO

    def test_terminar_justo_en_el_borde_no_suma_un_cuadrante(self):
        """
        Un municipio que acaba exactamente en lon -90 no entra en h09.

        Su última columna de píxeles es la que *termina* en el borde, y esa es
        del cuadrante de la izquierda. Contar el de la derecha significaría
        descargar 500 MB para leer cero píxeles.
        """
        coords = np.array([[-91.0, 14.0], [-90.0, 14.0], [-90.0, 15.0],
                           [-91.0, 15.0], [-91.0, 14.0]])
        assert cuadrantes_de_coordenadas(coords) == ["h08v07"]

    def test_el_antimeridiano_se_rechaza_en_vez_de_dar_la_vuelta_al_mundo(self):
        coords = np.array([[179.5, 10.0], [-179.5, 10.0], [-179.5, 11.0],
                           [179.5, 11.0], [179.5, 10.0]])
        with pytest.raises(ValueError, match="antimeridiano"):
            cuadrantes_de_coordenadas(coords)

    def test_el_caso_de_un_cuadrante_sigue_fallando_si_se_pide_uno(self):
        """`cuadrante_de_coordenadas` no puede devolver uno de cuatro en silencio."""
        with pytest.raises(ValueError, match="cruza de cuadrante"):
            cuadrante_de_coordenadas(cuadrado(ESQUINA, 1.0))


class TestReparto:
    """El corte en cuadrantes no crea ni destruye territorio."""

    FIGURAS = [
        ("dos_horizontal", cuadrado((-90.0, 15.0), 1.3), 2),
        ("dos_vertical", cuadrado((-95.0, 10.0), 1.3), 2),
        ("cuatro_esquina", cuadrado(ESQUINA, 1.3), 4),
        ("tres_en_ele", ele(ESQUINA, 1.6), 3),
        ("tres_en_triangulo", triangulo(ESQUINA, 1.6), 3),
        ("cuatro_rozando_en_diagonal", banda_diagonal(ESQUINA, 1.6), 4),
    ]

    @pytest.mark.parametrize("nombre,coords,n_piezas", FIGURAS)
    def test_las_piezas_suman_el_area_sin_cortar(self, nombre, coords, n_piezas):
        piezas = cobertura_por_cuadrante(coords, FORMA)
        area_piezas = sum(w for pesos in piezas.values() for _, _, w in pesos)
        pesos_globales, _, _ = cobertura_global(coords)
        assert area_piezas == pytest.approx(pesos_globales.sum(), rel=1e-9)

    @pytest.mark.parametrize("nombre,coords,n_piezas", FIGURAS)
    def test_solo_se_descarga_lo_que_tiene_pixeles(self, nombre, coords, n_piezas):
        """
        La envolvente es una cota superior, no la respuesta.

        Una figura en L o en diagonal tiene la caja en cuatro cuadrantes y
        píxeles en tres o en dos. Descargar los otros costaría cientos de megas
        para no aportar nada.
        """
        piezas = cobertura_por_cuadrante(coords, FORMA)
        assert len(piezas) == n_piezas
        assert set(piezas) <= set(cuadrantes_de_coordenadas(coords))
        assert all(pesos for pesos in piezas.values())

    @pytest.mark.parametrize("nombre,coords,n_piezas", FIGURAS)
    def test_ningun_pixel_esta_en_dos_piezas(self, nombre, coords, n_piezas):
        """
        El lado del cuadrante son píxeles enteros, así que el borde entre
        cuadrantes cae siempre en un borde de píxel: ninguno se parte.
        """
        piezas = cobertura_por_cuadrante(coords, FORMA)
        globales = {
            (c, x, y)
            for c, pesos in piezas.items()
            for x, y, _ in pesos
        }
        assert len(globales) == sum(len(p) for p in piezas.values())

    @pytest.mark.parametrize("nombre,coords,n_piezas", FIGURAS)
    def test_las_coordenadas_son_locales_a_su_cuadrante(self, nombre, coords, n_piezas):
        """Lo que se guarda es lo que indexa la imagen que se descarga."""
        alto, ancho = FORMA
        for pesos in cobertura_por_cuadrante(coords, FORMA).values():
            assert all(0 <= x < ancho and 0 <= y < alto for x, y, _ in pesos)
            assert all(0 < w <= 1 for _, _, w in pesos)

    def test_el_area_no_depende_de_donde_caiga_la_reticula(self):
        """
        La misma figura en el centro de un cuadrante y a caballo entre cuatro.

        Los dos centros están en un píxel entero de la retícula global, así que
        la geometría subpíxel es idéntica y el área tiene que serlo también: si
        el reparto perdiera o duplicara la franja de la costura, aquí se vería.
        """
        dentro = cobertura_por_cuadrante(cuadrado((-95.0, 15.0), 1.3), FORMA)
        cruzando = cobertura_por_cuadrante(cuadrado(ESQUINA, 1.3), FORMA)

        assert len(dentro) == 1 and len(cruzando) == 4
        area_dentro = sum(w for p in dentro.values() for _, _, w in p)
        area_cruzando = sum(w for p in cruzando.values() for _, _, w in p)
        assert area_cruzando == pytest.approx(area_dentro, rel=1e-12)

        # Y no solo el total: el multiconjunto de coberturas es el mismo, así
        # que tampoco cambia la ponderación de la frontera.
        pesos_dentro = sorted(w for p in dentro.values() for _, _, w in p)
        pesos_cruzando = sorted(w for p in cruzando.values() for _, _, w in p)
        assert pesos_cruzando == pytest.approx(pesos_dentro, rel=1e-12)

    def test_la_esquina_reparte_las_cuatro_partes_del_cuadrado(self):
        """Un cuadrado centrado en la esquina se parte en cuatro cuartos iguales."""
        piezas = cobertura_por_cuadrante(cuadrado(ESQUINA, 1.2), FORMA)
        areas = [sum(w for _, _, w in pesos) for pesos in piezas.values()]
        esperada = (1.2 * GRADO_PX) ** 2 / 4
        assert areas == pytest.approx([esperada] * 4, rel=1e-9)


class TestMarcoDeReferencia:
    def test_con_un_cuadrante_la_referencia_es_ese(self):
        assert cuadrante_referencia(["h08v07"]) == "h08v07"
        assert a_marco_referencia("h08v07", "h08v07", FORMA) == (0, 0)

    def test_la_referencia_es_la_esquina_noroeste(self):
        assert cuadrante_referencia(CUATRO) == "h08v07"
        assert a_marco_referencia("h09v08", "h08v07", FORMA) == (120, 120)
        assert a_marco_referencia("h09v07", "h08v07", FORMA) == (120, 0)


@pytest.fixture
def mosaico(tmp_path):
    """Los cuatro cuadrantes de la esquina, como archivos HDF5."""
    return {c: crear_cuadrante(tmp_path, c) for c in CUATRO}


def oraculo(coordenadas):
    """
    Métricas como si la imagen no estuviera cortada.

    Se calculan sobre la retícula global, sin saber nada de cuadrantes: la
    radianza de cada píxel se evalúa con la misma fórmula con la que se
    fabricaron los archivos, y la cobertura sale del polígono entero.
    """
    pesos, fila_0, columna_0 = cobertura_global(coordenadas)
    filas, columnas = np.nonzero(pesos)
    valores = np.array([
        radianza_sintetica(int(c) + columna_0, int(f) + fila_0)
        for f, c in zip(filas, columnas)
    ], dtype=float)
    cobertura = np.array([pesos[f, c] for f, c in zip(filas, columnas)], dtype=float)
    return metricas_ponderadas(valores, np.round(cobertura, 6))


class TestProcesamientoMulticuadrante:
    """Las métricas de un municipio repartido son las de la imagen sin cortar."""

    def test_cuatro_cuadrantes_dan_lo_mismo_que_una_imagen_entera(self, mosaico):
        coords = cuadrado(ESQUINA, 1.3)
        piezas = cobertura_por_cuadrante(coords, FORMA)

        r = process_image_mosaico(mosaico, piezas, date(2024, 1, 1), "sintetico",
                                  delete_files=False)
        esperado = oraculo(coords)

        assert r is not None
        assert r.Cuadrantes == CUATRO
        assert r.Cuadrante_referencia == "h08v07"
        assert r.Cantidad_de_pixeles == pytest.approx(esperado["Cantidad_de_pixeles"], rel=1e-12)
        assert r.Suma_de_radianza == pytest.approx(esperado["Suma_de_radianza"], rel=1e-12)
        assert r.Media_de_radianza == pytest.approx(esperado["Media_de_radianza"], rel=1e-12)
        assert r.Desviacion_estandar_de_radianza == pytest.approx(
            esperado["Desviacion_estandar_de_radianza"], rel=1e-12)
        # Los percentiles no son promediables: si se hubieran calculado por
        # cuadrante y mezclado después, aquí no coincidirían.
        assert r.Percentil_50_de_radianza == pytest.approx(
            esperado["Percentil_50_de_radianza"], rel=1e-12)
        assert r.Percentil_25_de_radianza == pytest.approx(
            esperado["Percentil_25_de_radianza"], rel=1e-12)
        assert r.Fraccion_valida == pytest.approx(1.0)

    @pytest.mark.parametrize("figura,n", [(cuadrado, 4), (ele, 3), (triangulo, 3)])
    def test_dos_tres_y_cuatro_cuadrantes(self, mosaico, figura, n):
        coords = figura(ESQUINA, 1.6)
        piezas = cobertura_por_cuadrante(coords, FORMA)
        assert len(piezas) == n

        r = process_image_mosaico(mosaico, piezas, date(2024, 1, 1), "sintetico",
                                  delete_files=False)
        esperado = oraculo(coords)
        assert r is not None
        assert len(r.Cuadrantes) == n
        assert r.Suma_de_radianza == pytest.approx(esperado["Suma_de_radianza"], rel=1e-12)
        assert r.Media_de_radianza == pytest.approx(esperado["Media_de_radianza"], rel=1e-12)

    def test_el_recorte_cruza_la_costura_sin_saltos(self, mosaico):
        """
        El recorte se publica en coordenadas del cuadrante de referencia, así
        que un municipio partido sale como una sola matriz continua. Se compara
        celda a celda contra la retícula global: un desplazamiento de un solo
        píxel en la composición rompería esta igualdad.
        """
        coords = cuadrado(ESQUINA, 1.3)
        piezas = cobertura_por_cuadrante(coords, FORMA)
        r = process_image_mosaico(mosaico, piezas, date(2024, 1, 1), "sintetico",
                                  delete_files=False)

        # El origen del marco de referencia (h08v07) en la retícula global.
        dx, dy = 8 * FORMA[1], 7 * FORMA[0]
        matriz = np.array(r.Matriz_de_radianza, dtype=float)
        assert matriz.shape == (r.Filas, r.Columnas)
        assert r.Columnas > FORMA[1] - r.Bbox.min_x  # el recorte cruza el borde

        for fila in range(r.Filas):
            for columna in range(r.Columnas):
                esperado = radianza_sintetica(
                    r.Bbox.min_x + columna + dx, r.Bbox.min_y + fila + dy
                )
                assert matriz[fila, columna] == pytest.approx(esperado)

        cobertura = np.array(r.Cobertura_municipio, dtype=float)
        assert cobertura.sum() == pytest.approx(r.Cantidad_de_pixeles, rel=1e-9)
        assert np.array(r.Mascara_municipio).sum() == sum(len(p) for p in piezas.values())

    def test_un_cuadrante_que_falta_se_declara_y_no_se_inventa(self, mosaico):
        """
        Si una de las cuatro imágenes no está, el territorio que cubría sale de
        la cuenta y `Fraccion_valida` lo dice. La alternativa —publicar el
        municipio con tres cuartos de su superficie y pinta de estar completo—
        es la que produce series que parecen válidas y no lo son.
        """
        coords = cuadrado(ESQUINA, 1.3)
        piezas = cobertura_por_cuadrante(coords, FORMA)
        rutas = dict(mosaico)
        rutas["h09v08"] = None

        r = process_image_mosaico(rutas, piezas, date(2024, 1, 1), "sintetico",
                                  delete_files=False)
        area_total = sum(w for p in piezas.values() for _, _, w in p)
        area_perdida = sum(w for _, _, w in piezas["h09v08"])

        assert r is not None
        assert r.Cuadrantes_faltantes == ["h09v08"]
        assert r.Fraccion_valida == pytest.approx(
            (area_total - area_perdida) / area_total, rel=1e-9)
        assert r.Cantidad_de_pixeles == pytest.approx(area_total - area_perdida, rel=1e-9)
        # El hueco queda como null en la matriz, no como cero: cero es una
        # medición de oscuridad, y esto es ausencia de medición.
        matriz = r.Matriz_de_radianza
        assert any(v is None for fila in matriz for v in fila)

    def test_un_cuadrante_sin_medicion_no_se_suma_como_radianza(self, tmp_path):
        """El relleno de un cuadrante no puede entrar como si fuera luz."""
        rutas = {c: crear_cuadrante(tmp_path, c, relleno=(c == "h09v08")) for c in CUATRO}
        coords = cuadrado(ESQUINA, 1.3)
        piezas = cobertura_por_cuadrante(coords, FORMA)

        r = process_image_mosaico(rutas, piezas, date(2024, 1, 1), "sintetico",
                                  delete_files=False)
        area_total = sum(w for p in piezas.values() for _, _, w in p)
        assert r is not None
        assert r.Fraccion_valida < 1.0
        assert r.Cantidad_de_pixeles < area_total

    def test_sin_ninguna_imagen_no_hay_medicion(self, tmp_path):
        piezas = cobertura_por_cuadrante(cuadrado(ESQUINA, 1.3), FORMA)
        assert process_image_mosaico({c: None for c in CUATRO}, piezas,
                                     date(2024, 1, 1), "sintetico") is None

    def test_un_cuadrante_mal_georreferenciado_no_contamina_el_mosaico(self, tmp_path):
        """
        Si un archivo dice estar en otro sitio, sus píxeles se descartan como si
        faltara. Componerlo igualmente pegaría radianza de otro lugar del planeta
        en el municipio.
        """
        rutas = {c: crear_cuadrante(tmp_path, c) for c in CUATRO}
        # h09v08 se sustituye por una imagen que declara ser h00v00.
        rutas["h09v08"] = crear_cuadrante(tmp_path, "h00v00")

        piezas = cobertura_por_cuadrante(cuadrado(ESQUINA, 1.3), FORMA)
        r = process_image_mosaico(rutas, piezas, date(2024, 1, 1), "sintetico",
                                  delete_files=False)
        assert r is not None
        assert r.Cuadrantes_faltantes == ["h09v08"]
        assert r.Fraccion_valida < 1.0


class TestUnCuadranteSigueIgual:
    """La generalización no puede cambiar lo que ya se publicaba."""

    def test_el_bbox_de_un_municipio_no_repartido_es_el_de_siempre(self, tmp_path):
        ruta = crear_cuadrante(tmp_path, "h08v07")
        coords = cuadrado((-95.0, 15.0), 1.3)
        piezas = cobertura_por_cuadrante(coords, FORMA)
        assert list(piezas) == ["h08v07"]

        r = process_image_mosaico({"h08v07": ruta}, piezas, date(2024, 1, 1),
                                  "sintetico", delete_files=False)
        xs = [x for x, _, _ in piezas["h08v07"]]
        ys = [y for _, y, _ in piezas["h08v07"]]
        assert (r.Bbox.min_x, r.Bbox.max_x) == (min(xs), max(xs))
        assert (r.Bbox.min_y, r.Bbox.max_y) == (min(ys), max(ys))
        assert r.Cuadrante_referencia == "h08v07"
        assert r.Cuadrantes_faltantes == []

    def test_las_metricas_de_un_cuadrante_coinciden_con_el_oraculo(self, tmp_path):
        ruta = crear_cuadrante(tmp_path, "h08v07")
        coords = cuadrado((-95.0, 15.0), 1.3)
        r = process_image_mosaico({"h08v07": ruta},
                                  cobertura_por_cuadrante(coords, FORMA),
                                  date(2024, 1, 1), "sintetico", delete_files=False)
        esperado = oraculo(coords)
        assert r.Media_de_radianza == pytest.approx(esperado["Media_de_radianza"], rel=1e-12)


class TestModeloDeCobertura:
    """El archivo de coberturas cambia de forma sin romper los que ya existen."""

    def test_lee_el_formato_de_un_cuadrante(self):
        datos = CoordenadasPixeles(cuadrante="h08v07", pesos=[(1, 1, 0.5), (2, 2, 1.0)])
        assert datos.cuadrantes == ["h08v07"]
        assert datos.cuadrante == "h08v07"
        assert datos.area == pytest.approx(1.5)

    def test_lee_el_formato_de_solo_coordenadas(self):
        datos = CoordenadasPixeles(cuadrante="h08v07", coordenadas_pixeles=[(1, 1), (2, 2)])
        assert datos.area == pytest.approx(2.0)
        assert datos.pesos == [(1, 1, 1.0), (2, 2, 1.0)]

    def test_lee_el_formato_de_varias_piezas(self):
        datos = CoordenadasPixeles(piezas=[
            {"cuadrante": "h08v07", "pesos": [(119, 5, 0.5)]},
            {"cuadrante": "h09v07", "pesos": [(0, 5, 0.5)]},
        ])
        assert datos.cuadrantes == ["h08v07", "h09v07"]
        assert datos.area == pytest.approx(1.0)

    def test_pedir_un_cuadrante_unico_a_un_municipio_repartido_falla(self):
        """
        Devolver el primero sería procesar media ciudad creyendo que está entera:
        exactamente el error que este trabajo viene a evitar.
        """
        datos = CoordenadasPixeles(piezas=[
            {"cuadrante": "h08v07", "pesos": [(119, 5, 0.5)]},
            {"cuadrante": "h09v07", "pesos": [(0, 5, 0.5)]},
        ])
        with pytest.raises(ValueError, match="ocupa 2 cuadrantes"):
            datos.cuadrante
        with pytest.raises(ValueError, match="ocupa 2 cuadrantes"):
            datos.pesos


class TestNoHayMedicionVsNoSePuedeMedir:
    """
    Dos huecos que antes eran el mismo hueco.

    `process_image_mosaico` atrapaba toda excepción y devolvía None, igual que
    cuando no había medición. Así, en una serie de diez años, una noche nublada
    y un defecto del código dejaban filas idénticamente ausentes. Ahora None
    significa «no hay medición y es normal», y `MedicionImposible` significa que
    la cobertura y las imágenes se contradicen.
    """

    def test_sin_imagen_ninguna_devuelve_none(self, tmp_path):
        """No hay imagen todavía: es un resultado, no un defecto."""
        coords = cuadrado(ESQUINA, 1.3)
        piezas = cobertura_por_cuadrante(coords, FORMA)
        r = process_image_mosaico({c: None for c in piezas}, piezas,
                                  date(2024, 1, 1), "sintetico", delete_files=False)
        assert r is None

    def test_todo_nublado_devuelve_none(self, tmp_path):
        """
        Todos los píxeles con la bandera de calidad en malo: hay imagen y hay
        territorio, pero ninguna observación utilizable. Sigue siendo un dato.
        """
        coords = cuadrado(ESQUINA, 1.3)
        piezas = cobertura_por_cuadrante(coords, FORMA)
        rutas = {c: crear_cuadrante(tmp_path, c, calidad_mala=True) for c in piezas}
        r = process_image_mosaico(rutas, piezas, date(2024, 1, 1), "sintetico",
                                  delete_files=False)
        assert r is None

    def test_la_tabla_sin_piezas_lanza(self):
        with pytest.raises(MedicionImposible, match="ninguna pieza"):
            process_image_mosaico({}, {}, date(2024, 1, 1), "sintetico",
                                  delete_files=False)

    def test_coordenadas_fuera_de_la_reticula_lanzan(self, tmp_path):
        """
        La tabla dice que el municipio está en un píxel que la imagen no tiene.
        Eso no se arregla esperando a mañana: es una incoherencia.
        """
        piezas = {"h08v07": [(9999, 9999, 1.0)]}
        rutas = {"h08v07": crear_cuadrante(tmp_path, "h08v07")}
        with pytest.raises(MedicionImposible, match="ninguna coordenada"):
            process_image_mosaico(rutas, piezas, date(2024, 1, 1), "sintetico",
                                  delete_files=False)

    def test_reticulas_incompatibles_lanzan(self, tmp_path):
        """Dos cuadrantes con distinto tamaño de píxel no se pueden componer."""
        coords = cuadrado(ESQUINA, 1.3)
        piezas = cobertura_por_cuadrante(coords, FORMA)
        rutas = {c: crear_cuadrante(tmp_path, c) for c in piezas}
        # Se rehace uno con el doble de resolución.
        raro = list(piezas)[-1]
        otro = tmp_path / "raro"
        otro.mkdir()
        rutas[raro] = crear_cuadrante(otro, raro, forma=(240, 240))

        with pytest.raises(MedicionImposible, match="retículas distintas"):
            process_image_mosaico(rutas, piezas, date(2024, 1, 1), "sintetico",
                                  delete_files=False)

    def test_un_defecto_inesperado_propaga(self, tmp_path, monkeypatch):
        """
        Antes se tragaba y salía como None. Un bug tiene que llegar a quien
        corre el pipeline, no disfrazarse de noche sin datos.
        """
        coords = cuadrado(ESQUINA, 1.3)
        piezas = cobertura_por_cuadrante(coords, FORMA)
        rutas = {c: crear_cuadrante(tmp_path, c) for c in piezas}

        def revienta(*a, **k):
            raise ValueError("un defecto cualquiera")

        monkeypatch.setattr("ntl.radianza.extraccion.metricas_ponderadas", revienta)
        with pytest.raises(ValueError, match="un defecto cualquiera"):
            process_image_mosaico(rutas, piezas, date(2024, 1, 1), "sintetico",
                                  delete_files=False)
