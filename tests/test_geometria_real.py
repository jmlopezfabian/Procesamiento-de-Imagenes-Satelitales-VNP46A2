"""
El multicuadrante, sobre geometría real.

Hasta aquí el soporte multicuadrante estaba probado con figuras sintéticas y con
municipios reales *trasladados* a las esquinas de la retícula. Las dos cosas son
necesarias —la sintética permite un oráculo exacto, la trasladada garantiza que
el área no cambia— pero ninguna usa un límite administrativo tal como lo publica
su fuente, cayendo donde de verdad cae.

Y eso importa porque los 18 municipios de `ntl_data/` caben todos en un cuadrante
y son polígonos simples: en producción, el camino nuevo no se ejecuta nunca. 21
de los 33 estados de México cruzan una línea de 10 grados. Los cinco de la
fixture cubren 1, 2, 3 y 4 cuadrantes, con y sin islas, con la geometría de
Natural Earth sin retocar.

Lo que se afirma es lo mismo de siempre, que es lo único que hay que afirmar: el
reparto en piezas es una partición exacta de lo que daría una retícula sin
cortar. Si eso se cumple sobre geometría real, las métricas se cumplen por
construcción, porque se calculan sobre los pares (radianza, cobertura) del
reparto.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import shape

from ntl.core.utils import cuadrantes_de_coordenadas
from ntl.geometria.cobertura import cobertura_exacta
from ntl.geometria.mosaico import (
    cobertura_por_cuadrante,
    cuadrante_referencia,
    partes,
    poligono_en_pixeles_globales,
)

FORMA = (2400, 2400)  # el tamaño real de un gránulo VNP46A2
FIXTURE = Path(__file__).parent / "fixtures" / "estados_multicuadrante.json"

# Lo que se espera de cada uno. Sale de la geometría publicada, no de correr el
# código: si un día Natural Earth cambia un límite, la prueba tiene que avisar.
ESPERADO = {
    "Distrito Federal": {"cuadrantes": 1, "partes": 1},
    "Quintana Roo": {"cuadrantes": 2, "partes": 5},
    "Guanajuato": {"cuadrantes": 3, "partes": 1},
    "Campeche": {"cuadrantes": 4, "partes": 2},
    "Sonora": {"cuadrantes": 4, "partes": 8},
}
# Intersecar exactamente un estado contra la retícula de 2400x2400 son segundos,
# no milisegundos, y cada prueba necesita el reparto y el oráculo. Guanajuato es
# el más barato de los que cruzan (4 s con oráculo), así que un cruce real se
# comprueba en cada corrida; el resto va marcado y lo ejecuta CI.
LENTOS = {"Sonora", "Campeche", "Quintana Roo"}

CASOS = [
    pytest.param(nombre, marks=pytest.mark.lento) if nombre in LENTOS else nombre
    for nombre in ESPERADO
]


@pytest.fixture(scope="module")
def estados():
    with open(FIXTURE, encoding="utf-8") as f:
        datos = json.load(f)
    return {x["properties"]["NOMGEO"]: shape(x["geometry"]) for x in datos["features"]}


@pytest.fixture(scope="module")
def coberturas(estados):
    """
    Reparto y oráculo de un estado, calculados a lo sumo una vez por módulo.

    Es perezoso a propósito: pedir Guanajuato no puede costar los 34 s de Sonora.
    """
    cache = {}

    def de(nombre):
        if nombre not in cache:
            geom = estados[nombre]
            cache[nombre] = (cobertura_por_cuadrante(geom, FORMA), _oraculo(geom), geom)
        return cache[nombre]

    return de


def _oraculo(geom):
    """
    La misma cobertura sobre la retícula global, sin cortar por cuadrantes.

    Redondea a 6 decimales y descarta los ceros *antes* de comparar, igual que
    `cobertura_por_cuadrante`. No es un ajuste para que cuadre: una esquina puede
    rozar una celda y dejar 1e-8 de cobertura, que al redondear queda en cero y
    no aporta ni área ni valor. Contarlo como píxel del municipio metería un
    registro que no aporta nada y ensuciaría `Cantidad_de_pixeles`. Comparar sin
    redondear hacía fallar justo a los dos MultiPolygon, por uno o dos píxeles de
    cobertura nula, con las áreas coincidiendo hasta el último decimal.
    """
    pesos, fila_0, columna_0 = cobertura_exacta(
        poligono_en_pixeles_globales(geom, FORMA))
    filas, columnas = np.nonzero(pesos)
    oraculo = {}
    for f, c in zip(filas, columnas):
        w = round(min(float(pesos[f, c]), 1.0), 6)
        if w > 0:
            oraculo[(int(c + columna_0), int(f + fila_0))] = w
    return oraculo


def _a_global(cuadrante, x, y):
    """Del par (cuadrante, píxel) de vuelta a la retícula global."""
    h, v = int(cuadrante[1:3]), int(cuadrante[4:6])
    return h * FORMA[1] + x, v * FORMA[0] + y


class TestGeometriaRealSinTrasladar:
    @pytest.mark.parametrize("nombre", list(ESPERADO))
    def test_cuantos_cuadrantes_y_cuantas_partes(self, estados, nombre):
        """
        Fija la forma de cada caso. Sonora entra aquí aunque sea lento: contar
        partes y cuadrantes no cuesta nada, y es lo que documenta que la fixture
        de verdad cubre 1, 2, 3 y 4 cuadrantes con y sin islas.
        """
        geom = estados[nombre]
        assert len(partes(geom)) == ESPERADO[nombre]["partes"]
        assert len(cuadrantes_de_coordenadas(geom)) >= ESPERADO[nombre]["cuadrantes"]

    @pytest.mark.parametrize("nombre", CASOS)
    def test_el_reparto_es_una_particion_exacta_de_la_reticula_global(
            self, coberturas, nombre):
        """
        La afirmación central: cada píxel del municipio aparece en exactamente
        una pieza, en el cuadrante que le toca, y con la misma cobertura que
        tendría si la imagen no estuviera cortada. Ni un píxel de más, ni uno de
        menos, ni uno con otro peso.
        """
        piezas, oraculo, _ = coberturas(nombre)

        repartido = {}
        for cuadrante, pesos in piezas.items():
            for x, y, w in pesos:
                clave = _a_global(cuadrante, x, y)
                assert clave not in repartido, f"{clave} aparece en dos piezas"
                repartido[clave] = w

        assert set(repartido) == set(oraculo), (
            f"{len(set(repartido) ^ set(oraculo))} píxeles de diferencia con la "
            f"retícula sin cortar")
        for clave, w in oraculo.items():
            assert repartido[clave] == pytest.approx(w, abs=1e-6)

    @pytest.mark.parametrize("nombre", CASOS)
    def test_el_area_no_la_cambia_el_corte(self, coberturas, nombre):
        piezas, oraculo, _ = coberturas(nombre)
        area_piezas = sum(w for pesos in piezas.values() for _, _, w in pesos)
        assert area_piezas == pytest.approx(sum(oraculo.values()), rel=1e-12)

    @pytest.mark.parametrize("nombre", CASOS)
    def test_cada_pieza_cae_dentro_de_su_cuadrante(self, coberturas, nombre):
        """
        Un índice fuera de rango no fallaría: numpy recorta por el otro extremo y
        devolvería píxeles de la otra punta de la imagen, con toda la pinta de
        ser los correctos.
        """
        piezas, _, _ = coberturas(nombre)
        alto, ancho = FORMA
        for cuadrante, pesos in piezas.items():
            for x, y, w in pesos:
                assert 0 <= x < ancho and 0 <= y < alto, \
                    f"{cuadrante} tiene un píxel en ({x}, {y})"
                assert 0 < w <= 1.0

    @pytest.mark.parametrize("nombre", CASOS)
    def test_no_se_descarga_lo_que_no_tiene_territorio(self, coberturas, nombre):
        """
        La envolvente puede tocar cuadrantes donde el estado no pone un solo
        píxel. Cada uno de esos son cientos de megas que no hay que bajar.
        """
        piezas, _, geom = coberturas(nombre)
        envolvente = cuadrantes_de_coordenadas(geom)
        assert set(piezas) <= set(envolvente)
        for cuadrante, pesos in piezas.items():
            assert pesos, f"{cuadrante} entró en el reparto sin píxeles"

    @pytest.mark.parametrize("nombre", CASOS)
    def test_el_cuadrante_de_referencia_es_el_del_noroeste(self, coberturas, nombre):
        piezas, _, _ = coberturas(nombre)
        referencia = cuadrante_referencia(list(piezas))
        h_ref, v_ref = int(referencia[1:3]), int(referencia[4:6])
        for cuadrante in piezas:
            assert int(cuadrante[1:3]) >= h_ref and int(cuadrante[4:6]) >= v_ref


@pytest.mark.lento
class TestSonora:
    """
    Los dos casos difíciles a la vez, sobre geometría real: cuatro cuadrantes y
    ocho partes. Las islas del Golfo de California caen en cuadrantes distintos
    del continente, que es exactamente la figura que rompía la regla de que una
    envolvente de cuatro obliga a tocar al menos tres.

    Son ~16 s de intersecciones exactas sobre millones de celdas, así que va
    aparte: `pytest -m lento`.
    """

    def test_el_reparto_es_una_particion_exacta(self, estados):
        geom = estados["Sonora"]
        piezas = cobertura_por_cuadrante(geom, FORMA)

        oraculo = _oraculo(geom)

        repartido = {}
        for cuadrante, pesos in piezas.items():
            for x, y, w in pesos:
                repartido[_a_global(cuadrante, x, y)] = w

        assert len(piezas) == 4
        assert set(repartido) == set(oraculo)
        assert sum(repartido.values()) == pytest.approx(sum(oraculo.values()), rel=1e-12)

    def test_las_islas_aportan_area_en_otros_cuadrantes(self, estados):
        """
        Si el lector se quedara con el primer anillo, como hacía antes, las islas
        no estarían y el área saldría más pequeña sin que nada lo dijera.
        """
        geom = estados["Sonora"]
        piezas = cobertura_por_cuadrante(geom, FORMA)
        continente = max(partes(geom), key=lambda p: p.area)

        area_total = sum(w for pesos in piezas.values() for _, _, w in pesos)
        area_continente = sum(
            w for pesos in cobertura_por_cuadrante(continente, FORMA).values()
            for _, _, w in pesos)

        assert area_total > area_continente
        assert len(partes(geom)) == 8
