"""
Excepciones del procesamiento.

La distinción que hace este módulo es la que separa un hueco legítimo en la
serie de uno que es un defecto. Sin ella, «esa noche estaba nublado» y «el
código se rompió» producen exactamente lo mismo —una fila que no está— y años
después no hay manera de decir cuál fue cuál.
"""


class MedicionImposible(Exception):
    """
    La medición no se puede calcular porque los datos de entrada no cuadran.

    No es lo mismo que no haya medición: que una noche esté nublada, o que la
    imagen todavía no exista en el archivo de la NASA, son resultados normales y
    se señalan devolviendo None. Esto es otra cosa: la tabla de cobertura y las
    imágenes se contradicen, y eso no se arregla esperando a mañana.
    """

    def __init__(self, motivo: str, municipio: str | None = None, fecha=None):
        self.motivo = motivo
        self.municipio = municipio
        self.fecha = fecha
        contexto = " en ".join(str(x) for x in (municipio, fecha) if x is not None)
        super().__init__(f"{motivo}" + (f" ({contexto})" if contexto else ""))
