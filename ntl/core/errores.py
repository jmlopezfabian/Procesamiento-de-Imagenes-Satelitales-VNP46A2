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


class CredencialesInvalidas(Exception):
    """
    LAADS rechazó el token: no tiene sentido reintentar ni seguir con el lote.

    Se levanta ante un 401/403 o ante una página HTML de login con código 200
    donde debía venir el HDF5. Es distinta de «el gránulo no existe» (que se
    señala con None): si el token está mal, todas las demás descargas van a
    fallar igual, y registrarlo como un fallo más por fecha convertiría una
    corrida entera en miles de huecos con la misma causa.
    """


class DescargaFallida(Exception):
    """
    Se agotaron los reintentos al listar o descargar: el gránulo puede existir.

    Distinta de «no existe» (404, cuadrante fuera del listado, día sin publicar),
    que es un resultado normal y se señala con None. Esto es un fallo transitorio
    que no se recuperó (503 persistente, red caída, cuerpo ilegible): si se
    tratara como falta de datos, un corte de NASA dejaría en la serie el mismo
    hueco que una noche sin imagen. Las funciones públicas `find_file_async` y
    `download_file_async` la traducen a None (su contrato no cambia); los lotes
    usan las variantes que la lanzan y la anotan en `fallos`.
    """

    def __init__(self, motivo: str):
        self.motivo = motivo
        super().__init__(motivo)
