from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Tuple
from datetime import date

class PiezaCuadrante(BaseModel):
    """
    La parte de un municipio que cae en un cuadrante concreto.

    `pesos` está en coordenadas de píxel **locales** a la retícula de ese
    cuadrante, que es como se indexa la imagen que se descarga.
    """

    cuadrante: str = Field(..., description="Cuadrante hHHvVV al que pertenecen los píxeles")
    pesos: List[Tuple[int, int, float]] = Field(
        ..., description="(x, y, coverage) locales al cuadrante; coverage en (0, 1]"
    )

    @property
    def area(self) -> float:
        return sum(w for _, _, w in self.pesos)


class CoordenadasPixeles(BaseModel):
    """
    Cobertura precalculada de un municipio, repartida entre los cuadrantes que toca.

    Cada entrada de `pesos` es (x, y, w) con w en (0, 1]: la fracción del píxel
    que cae dentro del municipio. Los píxeles de frontera valen menos de 1, que
    es justo lo que un conjunto de coordenadas no podía expresar.

    Un municipio no tiene por qué caber en un cuadrante: la retícula se corta
    cada 10 grados por conveniencia del archivo, no por ninguna frontera
    administrativa. Por eso la unidad es la lista de `piezas` y no un cuadrante
    único; el caso de una sola pieza, que es el habitual, se sigue escribiendo y
    leyendo como antes.
    """

    piezas: List[PiezaCuadrante] = Field(
        ..., description="Cobertura por cuadrante; normalmente una sola pieza"
    )

    @model_validator(mode="before")
    @classmethod
    def _acepta_formatos_anteriores(cls, data):
        """
        Lee los dos formatos previos: un cuadrante con pesos y un cuadrante con
        solo coordenadas.

        Un archivo viejo sigue cargando. El de solo coordenadas se lee con
        cobertura 1.0 en cada píxel, lo que reproduce el comportamiento
        anterior, que subestimaba el área del municipio porque descartaba la
        franja de frontera entera; regenera el archivo con
        scripts/generar_coordenadas_pixeles.py para corregirlo.
        """
        if not isinstance(data, dict) or "piezas" in data:
            return data

        data = dict(data)
        if "coordenadas_pixeles" in data and "pesos" not in data:
            data["pesos"] = [(x, y, 1.0) for x, y in data.pop("coordenadas_pixeles")]
        if "cuadrante" in data and "pesos" in data:
            data["piezas"] = [{"cuadrante": data.pop("cuadrante"),
                               "pesos": data.pop("pesos")}]
        return data

    @property
    def cuadrantes(self) -> List[str]:
        """Cuadrantes que el municipio toca, en orden de lectura del mosaico."""
        return [pieza.cuadrante for pieza in self.piezas]

    @property
    def cuadrante(self) -> str:
        """
        El cuadrante, cuando solo hay uno.

        Falla si el municipio ocupa varios en vez de devolver el primero: quien
        pida un cuadrante único a un municipio repartido está a punto de
        procesar media ciudad creyendo que la procesó entera.
        """
        if len(self.piezas) != 1:
            raise ValueError(
                f"El municipio ocupa {len(self.piezas)} cuadrantes "
                f"({', '.join(self.cuadrantes)}); usa .piezas o .cuadrantes."
            )
        return self.piezas[0].cuadrante

    @property
    def pesos(self) -> List[Tuple[int, int, float]]:
        """Los pesos, cuando hay un solo cuadrante. Ver `cuadrante`."""
        if len(self.piezas) != 1:
            raise ValueError(
                f"El municipio ocupa {len(self.piezas)} cuadrantes "
                f"({', '.join(self.cuadrantes)}); los pesos de cada uno son "
                f"locales a su retícula y no se pueden concatenar. Usa .piezas."
            )
        return self.piezas[0].pesos

    @property
    def coordenadas_pixeles(self) -> List[Tuple[int, int]]:
        """Solo las coordenadas, sin cobertura. Para máscaras y recortes."""
        return [(x, y) for x, y, _ in self.pesos]

    @property
    def area(self) -> float:
        """Área del municipio en píxeles de la retícula original, sumando piezas."""
        return sum(pieza.area for pieza in self.piezas)

class MetricasCapa(BaseModel):
    """
    Los mismos agregados que `MedicionResultado`, para una capa alternativa.

    Existe para que la capa rellenada se publique con exactamente las mismas
    estadísticas que la principal en vez de con un par de campos sueltos: quien
    compare las dos compara lo mismo.
    """

    Cantidad_de_pixeles: float = Field(..., description="Área en píxeles con valor en esta capa")
    Suma_de_radianza: float
    Media_de_radianza: float
    Desviacion_estandar_de_radianza: float
    Maximo_de_radianza: float
    Minimo_de_radianza: float
    Percentil_25_de_radianza: float
    Percentil_50_de_radianza: float
    Percentil_75_de_radianza: float
    Fraccion_valida: float = Field(
        ..., description="Fracción del área del municipio con valor en esta capa"
    )


class BboxRecorte(BaseModel):
    min_x: int = Field(..., description="Minimum x coordinate")
    max_x: int = Field(..., description="Maximum x coordinate")
    min_y: int = Field(..., description="Minimum y coordinate")
    max_y: int = Field(..., description="Maximum y coordinate")

class MedicionResultado(BaseModel):
    Fecha: date = Field(..., description="Date of the measurement")
    Municipio: Optional[str] = Field(None, description="Name of the municipality")
    Cantidad_de_pixeles: float = Field(
        ...,
        description=(
            "Area of the municipality in original pixels: the sum of the per-pixel "
            "coverage weights. Invariant to the scale factor, unlike a subpixel count."
        ),
    )
    Suma_de_radianza: float = Field(..., description="Sum of the radiance")
    Media_de_radianza: float = Field(..., description="Mean of the radiance")
    Desviacion_estandar_de_radianza: float = Field(..., description="Standard deviation of the radiance")
    Maximo_de_radianza: float = Field(..., description="Maximum of the radiance")
    Minimo_de_radianza: float = Field(..., description="Minimum of the radiance")
    Percentil_25_de_radianza: float = Field(..., description="25th percentile of the radiance")
    Percentil_50_de_radianza: float = Field(..., description="50th percentile of the radiance")
    Percentil_75_de_radianza: float = Field(..., description="75th percentile of the radiance")
    Producto: str = Field(
        "VNP46A2",
        description=(
            "Black Marble product the radiance came from. Always VNP46A2: "
            "BRDF/lunar-corrected with per-pixel quality flags. Series produced "
            "before this became mandatory used raw at-sensor radiance and their "
            "levels differ by 10-20%: do not mix them with these."
        ),
    )
    Unidades_de_radianza: str = Field(
        "nW/(cm2 sr)",
        description=(
            "Physical units of every radiance field. Series produced before the "
            "scale_factor was applied are in raw digital numbers and are 10x larger; "
            "they are NOT comparable with these values."
        ),
    )
    Fraccion_valida: float = Field(
        1.0,
        description=(
            "Fraction of the municipality area that had a valid reading. Below 1.0 the "
            "sum covers only part of the territory: the fill pixels were excluded "
            "instead of being counted as radiance."
        ),
    )
    Cuadrantes: List[str] = Field(
        default_factory=list,
        description=(
            "Tiles the municipality spans. More than one means the metrics were "
            "aggregated across images; an empty list means the field predates "
            "multi-tile support."
        ),
    )
    Cuadrantes_faltantes: List[str] = Field(
        default_factory=list,
        description=(
            "Tiles that could not be read for this date: their pixels entered the "
            "aggregate as NaN and were dropped, so this record covers only part of "
            "the municipality and Fraccion_valida is below 1. Filter on this field "
            "(or on Fraccion_valida) if your series does not admit partial records."
        ),
    )
    Cuadrante_referencia: Optional[str] = Field(
        None,
        description=(
            "Tile whose upper-left corner anchors Bbox and the cropped matrices: "
            "the north-westernmost of Cuadrantes. With a single tile it is that "
            "tile, so the coordinates mean what they always meant."
        ),
    )
    Bbox: Optional[BboxRecorte] = Field(
        None,
        description=(
            "Bounding box of the cropped image, in pixel coordinates of "
            "Cuadrante_referencia. Coordinates beyond the tile side (2400) fall "
            "in the tile to the east or south."
        ),
    )
    Filas: Optional[int] = Field(None, description="Number of rows in the cropped radiance matrix")
    Columnas: Optional[int] = Field(None, description="Number of columns in the cropped radiance matrix")
    Matriz_de_radianza: Optional[List[List[Optional[float]]]] = Field(
        None, description="Cropped radiance matrix (Filas x Columnas); NaN/Inf as null"
    )
    Mascara_municipio: Optional[List[List[int]]] = Field(
        None, description="Binary mask over the cropped matrix: 1=municipality pixel, 0=otherwise"
    )
    Cobertura_municipio: Optional[List[List[float]]] = Field(
        None,
        description=(
            "Fraction of each pixel of the cropped matrix that falls inside the "
            "municipality, in [0, 1]. Weight these to reproduce the aggregate metrics."
        ),
    )

    # --- Capa rellenada ---
    #
    # La NASA publica, junto a la medición, una capa que arrastra la última
    # recuperación buena donde no hubo ninguna. Da un valor todos los días, y
    # por eso mismo dos días seguidos sin recuperación traen el MISMO valor:
    # medido sobre Cuauhtémoc, el 1 y el 2 de septiembre de 2026 coinciden en
    # los 198 píxeles con diferencia 0.0.
    #
    # Se publica siempre con su antigüedad al lado. Un agregado de esta capa sin
    # `Antiguedad_mediana_dias` no dice si se midió esa noche o hace seis, y
    # ambos se grafican igual.
    Radianza_rellenada: Optional[MetricasCapa] = Field(
        None,
        description=(
            "Aggregates over Gap_Filled_DNB_BRDF-Corrected_NTL: the same value as the "
            "main algorithm where it retrieved, the last good retrieval carried "
            "forward where it did not. Present on days the main layer is empty, which "
            "is what makes a daily series continuous. Consecutive carried-forward days "
            "repeat the same observation: read Antiguedad_mediana_dias before treating "
            "them as independent."
        ),
    )
    Antiguedad_mediana_dias: Optional[float] = Field(
        None,
        description=(
            "Area-weighted median of Latest_High_Quality_Retrieval, in days. 0 means "
            "the filled layer was measured that night and equals the main layer; N "
            "means it is an observation from N days ago, repeated."
        ),
    )
    Fraccion_medida: Optional[float] = Field(
        None,
        description=(
            "Fraction of the municipality whose filled value was actually measured "
            "that night (age 0). The complement is carried forward. Filter on this "
            "for a series of genuine daily observations."
        ),
    )
    Matriz_rellenada: Optional[List[List[Optional[float]]]] = Field(
        None, description="Cropped Gap_Filled radiance matrix (Filas x Columnas)"
    )
    