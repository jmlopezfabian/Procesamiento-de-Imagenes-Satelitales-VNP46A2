import h5py
import math
import numpy as np
import os
from datetime import date
from typing import Any

from ..core.config import IMAGE_PATH, find_image_path
from ..core.errores import MedicionImposible
from ..core.lectura import leer_antiguedad, leer_radianza, leer_rellenada
from ..core.metricas import metricas_ponderadas
from ..core.models import MedicionResultado, MetricasCapa, PiezaCuadrante
from ..core.utils import verificar_georreferencia
from ..geometria.mosaico import a_marco_referencia, cuadrante_referencia


def _float_to_json_safe(value: float) -> float | None:
    """Convert float to JSON-serializable value; NaN and Inf become None."""
    if math.isfinite(value):
        return float(value)
    return None


def _normalizar_pesos(entradas) -> list[tuple[int, int, float]]:
    """
    Acepta tripletas (x, y, w) y también el formato anterior (x, y).

    Con el formato anterior no hay más remedio que asumir cobertura 1.0, que es
    justo lo que sesgaba las métricas: cada píxel de frontera cuenta entero
    aunque el municipio solo cubra la mitad. Se avisa en vez de fallar para que
    una tabla sin regenerar no rompa el procesamiento.
    """
    normalizados = []
    incompletas = 0
    for entrada in entradas:
        if len(entrada) == 3:
            x, y, w = entrada
        else:
            (x, y), w = entrada, 1.0
            incompletas += 1
        normalizados.append((int(x), int(y), float(w)))

    if incompletas:
        print(
            f"AVISO: {incompletas} píxeles sin cobertura; se asume 1.0. Las métricas "
            f"quedarán sesgadas por la frontera. Regenera la tabla con "
            f"scripts/generar_coordenadas_pixeles.py"
        )
    return normalizados


def _desplazamientos(cuadrantes: list[str | None], forma: tuple[int, int]
                     ) -> dict[str | None, tuple[int, int]]:
    """
    Origen de cada cuadrante en el marco del cuadrante de referencia.

    Con un solo cuadrante el desplazamiento es (0, 0) sea cual sea su
    identificador —incluso si no se conoce—, así que el recorte de un municipio
    que cabe en una imagen sigue expresado exactamente como siempre.
    """
    if len(cuadrantes) == 1:
        return {cuadrantes[0]: (0, 0)}
    referencia = cuadrante_referencia([c for c in cuadrantes])
    return {c: a_marco_referencia(c, referencia, forma) for c in cuadrantes}


def _crop_mosaico(
    matrices: dict[str | None, np.ndarray],
    pesos_ref: list[tuple[int, int, float]],
    desplazamientos: dict[str | None, tuple[int, int]],
    forma: tuple[int, int],
) -> dict[str, Any]:
    """
    Recorta la radianza al bounding box del municipio y construye, sobre ese
    recorte, la máscara binaria y la matriz de cobertura.

    Las coordenadas son las del cuadrante de referencia, así que un municipio
    repartido produce un recorte continuo aunque venga de varias imágenes: las
    columnas más allá del lado del cuadrante son las del cuadrante de al lado.
    Los huecos —un cuadrante que no se pudo leer— quedan como null, que es lo
    que son: territorio sin medición, no radianza cero.

    La máscara marca los píxeles que el municipio toca; la cobertura dice qué
    fracción de cada uno le corresponde, que es lo que pondera las métricas.
    """
    alto, ancho = forma
    min_x = min(x for x, _, _ in pesos_ref)
    max_x = max(x for x, _, _ in pesos_ref)
    min_y = min(y for _, y, _ in pesos_ref)
    max_y = max(y for _, y, _ in pesos_ref)
    filas, columnas = max_y - min_y + 1, max_x - min_x + 1

    radianza = np.full((filas, columnas), np.nan, dtype=float)
    for cuadrante, matriz in matrices.items():
        dx, dy = desplazamientos[cuadrante]
        # Intersección entre el recorte y el trozo de marco que ocupa este
        # cuadrante, en coordenadas del marco de referencia.
        x0, x1 = max(min_x, dx), min(max_x + 1, dx + ancho)
        y0, y1 = max(min_y, dy), min(max_y + 1, dy + alto)
        if x0 >= x1 or y0 >= y1:
            continue
        radianza[y0 - min_y : y1 - min_y, x0 - min_x : x1 - min_x] = matriz[
            y0 - dy : y1 - dy, x0 - dx : x1 - dx
        ]

    mask = np.zeros((filas, columnas), dtype=int)
    coverage = np.zeros((filas, columnas), dtype=float)
    for x, y, w in pesos_ref:
        mask[y - min_y, x - min_x] = 1
        coverage[y - min_y, x - min_x] = w

    radiance_list: list[list[float | None]] = [
        [_float_to_json_safe(float(v)) for v in row] for row in radianza
    ]

    return {
        "bbox": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
        "rows": filas,
        "cols": columnas,
        "radiance_matrix": radiance_list,
        "municipality_mask": [list(row) for row in mask],
        "municipality_coverage": [[float(v) for v in row] for row in coverage],
    }


def _crop_radiance_and_mask(
    image_matrix: np.ndarray, pesos_validos: list[tuple[int, int, float]]
) -> dict[str, Any]:
    """Recorte de un municipio que cabe entero en una imagen. Ver `_crop_mosaico`."""
    return _crop_mosaico(
        {None: image_matrix}, pesos_validos, {None: (0, 0)}, image_matrix.shape
    )


def extract_radiance_matrix_mosaico(
    rutas_por_cuadrante: dict[str | None, str | None],
    piezas,
    date_obj: date,
    municipio: str,
) -> dict[str, Any] | None:
    """
    Submatriz de radianza y máscara del municipio, compuestas entre cuadrantes.

    Las coordenadas del recorte son las del cuadrante de referencia (el del
    extremo noroeste), así que el municipio sale continuo aunque venga de varias
    imágenes. Los cuadrantes que no se pudieron leer quedan como null en la
    matriz, no como cero.
    """
    piezas_norm = _normalizar_piezas(piezas)
    if not piezas_norm:
        raise MedicionImposible(
            "la tabla de cobertura no trae ninguna pieza", municipio, date_obj)

    matrices: dict[str | None, np.ndarray] = {}
    for cuadrante, _ in piezas_norm.items():
        ruta = rutas_por_cuadrante.get(cuadrante)
        if not ruta or not _es_hdf5(ruta):
            continue
        try:
            with h5py.File(ruta, "r") as hdf_file:
                image_matrix = hdf_file[find_image_path(hdf_file)][()]
                if cuadrante:
                    verificar_georreferencia(hdf_file, cuadrante, image_matrix.shape)
            matrices[cuadrante] = image_matrix
        except Exception as e:
            print(f"Error extrayendo matriz de {ruta}: {e}")

    if not matrices:
        print(f"No se pudo leer ningún cuadrante de {municipio} en {date_obj}")
        return None

    sin_imagen = [c for c in piezas_norm if c and c not in matrices]
    if sin_imagen:
        print(f"⚠️ {municipio} en {date_obj}: sin imagen para {', '.join(sin_imagen)}. "
              f"Esa parte del recorte sale como null (NaN), no como cero: es "
              f"territorio sin medición, no oscuridad.")

    formas = {m.shape for m in matrices.values()}
    if len(formas) > 1:
        raise MedicionImposible(
            f"los cuadrantes traen retículas distintas {formas} y no se pueden "
            f"componer", municipio, date_obj)
    forma = formas.pop()
    alto, ancho = forma
    desplazamientos = _desplazamientos(list(piezas_norm), forma)

    pesos_ref = [
        (x + desplazamientos[c][0], y + desplazamientos[c][1], w)
        for c, pesos in piezas_norm.items()
        for x, y, w in pesos
        if 0 <= y < alto and 0 <= x < ancho
    ]
    if not pesos_ref:
        raise MedicionImposible(
            "ninguna coordenada de la tabla de cobertura cae dentro de la "
            "retícula", municipio, date_obj)

    cuadrantes = [c for c in piezas_norm if c]
    return {
        "municipio": municipio,
        "fecha": date_obj,
        "cuadrantes": cuadrantes,
        "cuadrantes_faltantes": [c for c in cuadrantes if c not in matrices],
        "cuadrante_referencia": cuadrante_referencia(cuadrantes) if cuadrantes else None,
        **_crop_mosaico(matrices, pesos_ref, desplazamientos, forma),
    }


def extract_radiance_matrix(
    downloaded_path: str,
    pesos: list[tuple[int, int, float]],
    date_obj: date,
    municipio: str,
    cuadrante: str | None = None,
) -> dict[str, Any] | None:
    """
    Extrae la submatriz de radianza y la máscara binaria del municipio, recortadas al bounding box.
    Devuelve dict con radiance_matrix, municipality_mask, bbox, rows, cols, municipio, fecha.

    `cuadrante` es opcional solo por compatibilidad: si se pasa, se comprueba
    que el archivo esté georreferenciado donde la tabla de coberturas supone.

    Caso de un solo cuadrante de `extract_radiance_matrix_mosaico`.
    """
    return extract_radiance_matrix_mosaico(
        {cuadrante: downloaded_path}, {cuadrante: pesos}, date_obj, municipio
    )


def _es_hdf5(ruta: str) -> bool:
    """Descarta el HTML que LAADS devuelve cuando la sesión caducó."""
    if not os.path.exists(ruta):
        print(f"Archivo no encontrado: {ruta}")
        return False
    with open(ruta, "rb") as f:
        start = f.read(15)
    if b"<html" in start or b"<!DOCTYPE html" in start:
        print(f"Archivo HTML recibido en vez de HDF5: {ruta}")
        return False
    return True


def _metricas_vacias() -> dict:
    """
    Agregados de una capa que no midió nada: área cero, no valores inventados.

    Se usa cuando el algoritmo principal no recuperó ningún píxel pero la capa
    rellenada sí trae valor. El registro tiene que decir «esta noche no se
    midió», y un cero de área lo dice; copiar ahí el relleno lo borraría.
    """
    return {
        "Cantidad_de_pixeles": 0.0,
        "Suma_de_radianza": 0.0,
        "Media_de_radianza": 0.0,
        "Desviacion_estandar_de_radianza": 0.0,
        "Maximo_de_radianza": 0.0,
        "Minimo_de_radianza": 0.0,
        "Percentil_25_de_radianza": 0.0,
        "Percentil_50_de_radianza": 0.0,
        "Percentil_75_de_radianza": 0.0,
        "Fraccion_valida": 0.0,
    }


def _resumir_antiguedad(edades: np.ndarray, pesos: np.ndarray):
    """
    (mediana ponderada de la antigüedad, fracción de área medida esa noche).

    La mediana se pondera por área igual que el resto de las métricas, para que
    un píxel de frontera a medias no pese como uno entero. La fracción medida
    cuenta el área con antigüedad 0: es la parte del municipio en la que la
    capa rellenada vale una observación de esa noche y no una repetida.
    """
    finitas = np.isfinite(edades) & (pesos > 0)
    if not finitas.any():
        return None, None

    e, w = edades[finitas], pesos[finitas]
    orden = np.argsort(e)
    e, w = e[orden], w[orden]
    acumulado = np.cumsum(w)
    mediana = float(e[np.searchsorted(acumulado, acumulado[-1] / 2.0)])

    area_total = float(pesos[pesos > 0].sum())
    medida = float(w[e == 0].sum()) / area_total if area_total > 0 else 0.0
    return mediana, medida


def _crop_capa(capas, pesos_ref, desplazamientos, forma):
    """Recorte de una capa alternativa al mismo bbox que el principal, o None."""
    presentes = {c: m for c, m in capas.items() if m is not None}
    if not presentes:
        return None
    return _crop_mosaico(presentes, pesos_ref, desplazamientos, forma)["radiance_matrix"]


def _leer_cuadrante(ruta: str, cuadrante: str | None):
    """
    Las capas de un cuadrante, en unidades físicas y con NaN donde no hay medición.

    Devuelve (matriz, lectura) o None si el archivo no sirve. `lectura` lleva
    además la capa rellenada y la antigüedad cuando el gránulo las trae, que es
    lo que permite emitir un registro las noches en que el algoritmo principal
    no recuperó nada.

    No lanza: en un mosaico, que falte un cuadrante no debe tirar el municipio
    entero; el territorio perdido se contabiliza después en Fraccion_valida.
    """
    try:
        if not _es_hdf5(ruta):
            return None
        with h5py.File(ruta, "r") as hdf_file:
            image_matrix, lectura = leer_radianza(hdf_file)
            if cuadrante:
                verificar_georreferencia(hdf_file, cuadrante, image_matrix.shape)
            rellenada = leer_rellenada(hdf_file)
            lectura = dict(lectura)
            lectura["rellenada"] = rellenada[0] if rellenada else None
            lectura["antiguedad"] = leer_antiguedad(hdf_file)
        return image_matrix, lectura
    except Exception as e:
        print(f"Error leyendo {cuadrante or ruta}: {e}")
        return None


def _normalizar_piezas(piezas) -> dict[str | None, list[tuple[int, int, float]]]:
    """Acepta lista de PiezaCuadrante o dict {cuadrante: pesos}."""
    if isinstance(piezas, dict):
        return {c: _normalizar_pesos(p) for c, p in piezas.items()}
    return {pieza.cuadrante: _normalizar_pesos(pieza.pesos) for pieza in piezas}


def process_image_mosaico(rutas_por_cuadrante, piezas, date_obj, municipio,
                          delete_files=True):
    """
    Métricas de un municipio repartido entre varios cuadrantes.

    Un municipio no tiene por qué caber en un cuadrante: la retícula se corta
    cada 10 grados por conveniencia del archivo, no por ninguna frontera
    administrativa. Antes ese municipio ni siquiera entraba en la tabla de
    cobertura; procesarlo con el cuadrante que más territorio le tocara habría
    sido peor, porque el resto se descartaba en silencio y la media salía de
    media ciudad.

    Las métricas no se calculan por cuadrante y se promedian: se junta el par
    (radianza, cobertura) de todos los píxeles del municipio, vengan de la
    imagen que vengan, y se agrega **una sola vez**. Así el resultado es
    idénticamente el que daría una imagen sin cortar, incluidos los percentiles,
    que no son promediables.

    Si un cuadrante no se puede leer, sus píxeles entran como **NaN**: el área
    que aportaban sale de la cuenta y `Fraccion_valida` lo deja por escrito, en
    vez de publicar un municipio recortado con pinta de estar completo. El
    registro se produce igualmente, parcial y marcado como tal —se avisa por
    consola con el porcentaje de área perdida—; una serie que no admita
    registros parciales debe filtrarlos por `Fraccion_valida` o por
    `Cuadrantes_faltantes`, que para eso están.

    `None` y excepción no son lo mismo, y esa es la distinción que sostiene la
    serie histórica. Devuelve **None** cuando no hay medición y eso es normal:
    ninguna imagen disponible todavía, o ningún píxel con observación válida
    porque la noche estaba nublada. **Lanza `MedicionImposible`** cuando la tabla
    de cobertura y las imágenes se contradicen, que no se arregla esperando a
    mañana, y deja **propagar** cualquier otra excepción, que es un defecto.
    Antes las tres cosas se atrapaban aquí y salían como None: un hueco por nubes
    y un hueco por bug eran indistinguibles a los dos años.

    Args:
        rutas_por_cuadrante: {cuadrante: ruta del HDF5 descargado, o None}
        piezas: lista de PiezaCuadrante, o {cuadrante: [(x, y, w), ...]}
        date_obj: Fecha de la medición
        municipio: Nombre del municipio
        delete_files: Si borrar los HDF5 al terminar

    Returns:
        MedicionResultado, o None si no hay medición que calcular.

    Raises:
        MedicionImposible: la cobertura y las imágenes no cuadran.
    """
    piezas_norm = _normalizar_piezas(piezas)
    if not piezas_norm:
        raise MedicionImposible(
            "la tabla de cobertura no trae ninguna pieza", municipio, date_obj)

    try:
        matrices: dict[str | None, np.ndarray] = {}
        rellenadas: dict[str | None, np.ndarray | None] = {}
        antiguedades: dict[str | None, np.ndarray | None] = {}
        lectura = None
        for cuadrante in piezas_norm:
            ruta = rutas_por_cuadrante.get(cuadrante)
            leido = _leer_cuadrante(ruta, cuadrante) if ruta else None
            if leido is None:
                continue
            matrices[cuadrante], lectura_cuadrante = leido
            rellenadas[cuadrante] = lectura_cuadrante.get("rellenada")
            antiguedades[cuadrante] = lectura_cuadrante.get("antiguedad")
            lectura = lectura or lectura_cuadrante

        faltantes = [c for c in piezas_norm if c not in matrices]
        if not matrices:
            print(f"⚠️ {municipio} en {date_obj}: no se pudo leer ninguno de los "
                  f"cuadrantes {list(piezas_norm)}")
            return None
        if faltantes:
            # Se dice el tamaño del agujero antes de agregar nada. Quien lea el
            # registro después ve Fraccion_valida, pero quien mira la corrida
            # tiene que poder decidir en el momento si esa fecha le sirve.
            area_total = sum(w for pesos in piezas_norm.values() for _, _, w in pesos)
            area_faltante = sum(w for c in faltantes for _, _, w in piezas_norm[c])
            porcentaje = 100 * area_faltante / area_total if area_total else 0.0
            print(f"⚠️ {municipio} en {date_obj}: sin imagen para "
                  f"{', '.join(faltantes)}. Sus píxeles entran como NaN y quedan "
                  f"fuera del agregado: es el {porcentaje:.1f}% del área del "
                  f"municipio ({area_faltante:.2f} de {area_total:.2f} px). El "
                  f"registro sale parcial, con Fraccion_valida ≈ "
                  f"{1 - area_faltante / area_total if area_total else 0:.4f} y "
                  f"Cuadrantes_faltantes={faltantes}. Si tu serie no admite "
                  f"registros parciales, descártalo por ese campo.")

        formas = {m.shape for m in matrices.values()}
        if len(formas) > 1:
            raise MedicionImposible(
                f"los cuadrantes traen retículas distintas {formas} y no se "
                f"pueden componer", municipio, date_obj)
        forma = formas.pop()
        alto, ancho = forma

        desplazamientos = _desplazamientos(list(piezas_norm), forma)

        # Los pesos llegan locales a cada cuadrante; el recorte los necesita en
        # un marco común, y la radianza en el de su propia imagen.
        valores, cobertura, pesos_ref = [], [], []
        valores_rellenados, edades = [], []
        for cuadrante, pesos in piezas_norm.items():
            dx, dy = desplazamientos[cuadrante]
            matriz = matrices.get(cuadrante)
            relleno = rellenadas.get(cuadrante)
            edad = antiguedades.get(cuadrante)
            fuera = 0
            for x, y, w in pesos:
                if not (0 <= y < alto and 0 <= x < ancho):
                    fuera += 1
                    continue
                # Un cuadrante que falta aporta área sin medición: NaN, no cero.
                valores.append(float(matriz[y, x]) if matriz is not None else np.nan)
                valores_rellenados.append(
                    float(relleno[y, x]) if relleno is not None else np.nan)
                edades.append(float(edad[y, x]) if edad is not None else np.nan)
                cobertura.append(w)
                pesos_ref.append((x + dx, y + dy, w))
            if fuera:
                print(f"ℹ️ {municipio}: {fuera} píxeles de {cuadrante} caen fuera "
                      f"de la retícula {forma} y se descartan")

        if not pesos_ref:
            raise MedicionImposible(
                "ninguna coordenada de la tabla de cobertura cae dentro de la "
                "retícula", municipio, date_obj)

        pesos_np = np.array(cobertura, dtype=float)
        metricas = metricas_ponderadas(np.array(valores), pesos_np)
        metricas_relleno = metricas_ponderadas(np.array(valores_rellenados), pesos_np)
        edad_mediana, fraccion_medida = _resumir_antiguedad(np.array(edades), pesos_np)

        # Antes se devolvía None en cuanto el algoritmo principal no recuperaba
        # nada, y esa noche desaparecía de la serie. Pero el gránulo sí trae
        # algo: la capa rellenada. Sobre Cuauhtémoc, 13 de 16 fechas publicadas
        # de una ventana de tres semanas no tenían ninguna recuperación y aun
        # así traían valor rellenado. Descartarlas era tirar lo que el satélite
        # sí entregó.
        #
        # Se emite registro si CUALQUIERA de las dos capas trae algo. Las dos
        # viajan etiquetadas —Fraccion_valida para la medida, Antiguedad y
        # Fraccion_medida para la rellenada—, así que quien consuma decide.
        if metricas is None and metricas_relleno is None:
            print(f"⚠️ {municipio} en {date_obj}: ninguna capa del gránulo trae "
                  f"valor para este municipio")
            return None

        if metricas is None:
            print(f"ℹ️ {municipio} en {date_obj}: sin recuperación del algoritmo "
                  f"principal; el registro sale solo con la capa rellenada "
                  f"(antigüedad mediana {edad_mediana:.0f} d). No es una medición "
                  f"de esta noche.")
            # Los agregados de la capa principal quedan en cero-área, no
            # inventados a partir del relleno: el registro dice que no se midió.
            metricas = _metricas_vacias()
        elif metricas["Fraccion_valida"] < 1.0:
            print(f"⚠️ {municipio} en {date_obj}: solo "
                  f"{metricas['Fraccion_valida']*100:.1f}% del territorio trae medición")

        crop = _crop_mosaico(matrices, pesos_ref, desplazamientos, forma)
        cuadrantes = [c for c in piezas_norm if c]

        return MedicionResultado(
            Fecha=date_obj,
            Municipio=municipio,
            Producto=lectura["producto"],
            Unidades_de_radianza=lectura["unidades"] or "nW/(cm2 sr)",
            **metricas,
            Cuadrantes=cuadrantes,
            Cuadrantes_faltantes=[c for c in faltantes if c],
            Cuadrante_referencia=cuadrante_referencia(cuadrantes) if cuadrantes else None,
            Bbox=crop["bbox"],
            Filas=crop["rows"],
            Columnas=crop["cols"],
            Matriz_de_radianza=crop["radiance_matrix"],
            Mascara_municipio=crop["municipality_mask"],
            Cobertura_municipio=crop["municipality_coverage"],
            Radianza_rellenada=(
                MetricasCapa(**metricas_relleno) if metricas_relleno else None),
            Antiguedad_mediana_dias=edad_mediana,
            Fraccion_medida=fraccion_medida,
            Matriz_rellenada=_crop_capa(
                rellenadas, pesos_ref, desplazamientos, forma),
        )

    finally:
        if delete_files:
            for ruta in set(r for r in rutas_por_cuadrante.values() if r):
                try:
                    if os.path.exists(ruta):
                        os.remove(ruta)
                        print(f"Archivo eliminado: {ruta}")
                except Exception as e:
                    print(f"Error eliminando archivo {ruta}: {e}")


def process_image(downloaded_path, pesos, date_obj, municipio, delete_file=True,
                  cuadrante=None):
    """
    Calcula las métricas de un municipio que cabe entero en un cuadrante.

    `pesos` son tripletas (x, y, w) con w en (0, 1]. Contar los píxeles enteros
    subestimaba el área entre 7% y 31% según la forma del municipio, porque
    descartaba una franja de frontera que está cubierta a medias.

    Si se pasa `cuadrante`, antes de tocar la radianza se comprueba que el
    archivo declare la misma esquina que se usó para construir las coberturas.
    Sin esa comprobación, un cambio de convención en el producto desplazaría
    todos los municipios en silencio; un solo píxel de desalineamiento mueve la
    media municipal un 4.6% en la mediana de las alcaldías.

    Caso particular de `process_image_mosaico` con una sola pieza; para un
    municipio repartido entre cuadrantes hay que usar aquella, porque aquí solo
    entra una imagen y el resto del territorio se perdería.
    """
    return process_image_mosaico(
        {cuadrante: downloaded_path},
        {cuadrante: pesos},
        date_obj,
        municipio,
        delete_files=delete_file,
    )
