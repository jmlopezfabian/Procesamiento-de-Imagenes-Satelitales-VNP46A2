## Procesamiento de Imágenes Satelitales Black Marble

Proyecto para procesar imágenes satelitales de luminosidad nocturna del producto **VNP46A2**
(Black Marble, NASA) y obtener métricas de radianza por municipio.

Es una **librería**: el tablero y el servicio que la usan a diario viven en
daily-ntl. Ver "Dónde vive la aplicación".

### Componentes principales

El paquete `ntl/` está dividido por responsabilidad, no por modelo de concurrencia:

- `ntl/core/`: configuración, modelos, utilidades y descargas.
- `ntl/geometria/`: polígono municipal → cobertura por píxel. Corre **una vez por
  municipio**; su resultado es estático mientras no cambie la delimitación oficial.
- `ntl/radianza/`: cobertura → métricas de luminosidad. Corre **todos los días**
  sobre cada imagen, descargando en paralelo y agrupando por cuadrante.
- `ntl_data/`: datos auxiliares (tabla de cobertura, límites geográficos).

El paquete se llama `ntl` por *nighttime lights*, el acrónimo con el que la literatura
nombra el fenómeno y que el reporte técnico ya usa como notación ($NTL_{i,j}$). No lleva
el identificador del producto de la NASA para no atarse a él: VNP46A1 es hoy la fuente,
pero el procesamiento no depende de que lo siga siendo.

`geometria` produce lo que `radianza` consume. La distinción entre síncrono y asíncrono
vive únicamente en `core/downloader.py`, donde las funciones asíncronas llevan el sufijo
`_async`.

Se utilizan **Pydantic v2** y modelos como `MedicionResultado` para validar y serializar los resultados.

---

## Requisitos e instalación

- Python 3.11+ (recomendado 3.12)
- `pip` y `virtualenv` (o equivalente)

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
# Para desarrollo y tests:
pip install -r requirements-dev.txt
```



## Dónde vive la aplicación

Este repositorio es una **librería**. Responde una pregunta: *dado un polígono y
una fecha, ¿cuánta radianza hubo?* No sabe qué es HTTP, una base de datos, un
usuario ni un calendario, y nada aquí importa desde la aplicación.

El tablero, la API, la caché, la autenticación y la recolección diaria están en
**daily-ntl**, que instala este paquete como dependencia. La flecha va en un solo
sentido y conviene que siga así: cuánto esperar a que la NASA publique es política
de operación, no un hecho sobre cómo se calcula la radianza.

Hasta la versión 0.8.0 este repositorio traía además un `api/` con su propio
servidor FastAPI, un agente y un `index.html`. Era una demostración —el
empaquetado ya la excluía de la distribución y la escondía tras un extra
`[api]`— pero duplicaba a daily-ntl con jobs en memoria, sin caché persistente y
sin autenticación. Se quitó para que no hubiera dos aplicaciones compitiendo por
el mismo trabajo. Lo único que tenía y daily-ntl no, el agente conversacional,
se movió allá.

Para usar el procesamiento desde código, ver "Ejemplos de uso desde código" más
abajo.

---

## Flujo de procesamiento (vista rápida)

![Flujo de píxeles y huecos](images/Flujo_pixeles_hueco.PNG)

1. Para cada fecha y municipio:
   - Se descarga el archivo HDF5 VNP46A1 correspondiente (NASA).
   - Se recorta la imagen usando la tabla de cobertura del municipio.
2. Se calculan métricas de radianza **ponderando cada píxel por la fracción que el
   municipio cubre de él**, y se modelan con `MedicionResultado`.
3. Los resultados se consolidan en un `DataFrame` de `pandas` y se **guardan como Parquet** para análisis posterior.

`Cantidad_de_pixeles` es el **área del municipio en píxeles**, no un conteo: es la suma
de las coberturas, así que puede ser fraccionaria. Contar píxeles enteros obligaba a
aceptar o descartar cada celda de frontera, y como esas celdas están cubiertas
aproximadamente por la mitad, descartarlas subestimaba el área entre 7% y 31% según la
forma del municipio.

La tabla de cobertura (`ntl_data/municipios_coordenadas_pixeles.json`) se regenera
con `python scripts/generar_coordenadas_pixeles.py`; solo hace falta si cambian los
polígonos municipales. La cobertura se calcula por intersección geométrica exacta,
sin factor de subdivisión ni pesos que elegir.

### Producto: VNP46A2 por omisión

Se lee **VNP46A2**, la escena corregida por BRDF lunar y atmósfera, con
`Mandatory_Quality_Flag` por píxel. **Es el único producto soportado.** Un archivo que
no traiga la bandera de calidad se rechaza en vez de procesarse sin filtrar.

VNP46A1 —radianza cruda al sensor— quedó fuera a propósito. No trae ninguna bandera que
permita descartar una observación inservible: una noche completamente nublada produce un
número plausible que no es luz del suelo. Dejarlo disponible como opción invitaba a
producir series que parecen válidas y no lo son.

El efecto sobre la serie es grande. Variación día a día sobre cinco fechas de 2025:

| Municipio | VNP46A1 | VNP46A2 |
|---|---|---|
| Iztapalapa | 12.1% | **3.8%** |
| Azcapotzalco | 12.8% | **3.4%** |
| Milpa Alta | 23.9% | **12.0%** |

En una de esas cinco fechas, el 5 de enero, VNP46A2 no tiene **ni un píxel utilizable**
en Iztapalapa: estaba 100% nublado. VNP46A1 entregaba 62,838 como si fuera una medición.

Las series generadas antes de este cambio usan VNP46A1 y sus niveles difieren entre 10% y
20%: **no se pueden mezclar con las nuevas**. Cada registro trae `Producto` para poder
distinguirlas.

### Unidades: aviso importante

Los valores de radianza están en **nW/(cm² sr)**, aplicando el `scale_factor` que
declara el producto. Las series generadas **antes de septiembre de 2026** están en
cuentas digitales sin escalar y son **diez veces mayores**: no se pueden concatenar
con las nuevas sin convertir. Cada registro trae `Unidades_de_radianza` para poder
distinguirlas.

Los píxeles sin medición (`_FillValue`) se excluyen del agregado en vez de sumarse
como radianza. `Fraccion_valida` indica qué parte del territorio del municipio traía
dato: por debajo de 1.0, la suma cubre solo esa fracción. Las series anteriores
incluyen 20 registros contaminados —cuatro fechas en las que el cuadrante entero
vino vacío— que se retiran con `python scripts/purgar_registros_invalidos.py`.

La API usa `ntl.radianza`, que descarga de forma asíncrona y agrupa por cuadrante, para mejorar el rendimiento cuando se procesan muchas fechas o municipios.

### Municipios repartidos entre varios cuadrantes

La retícula de Black Marble se corta cada 10 grados por conveniencia del archivo,
no por ninguna frontera administrativa. Un municipio pegado a un múltiplo de 10 en
longitud cae en dos cuadrantes, en latitud cae en dos, y cerca de una esquina de la
retícula cae en cuatro. Esos municipios se procesan componiendo las imágenes:

- La cobertura se calcula una vez sobre una **retícula global** anclada en (-180, 90)
  y se reparte en piezas, una por cuadrante. Como el lado del cuadrante son 2400
  píxeles enteros, ningún píxel se parte entre dos imágenes: la suma de las áreas de
  las piezas es exactamente el área sin cortar.
- Las métricas **no** se promedian entre cuadrantes. Se juntan los pares
  (radianza, cobertura) de todos los píxeles y se agregan una sola vez, que es la
  única forma de que los percentiles signifiquen algo.
- `Cuadrantes` lista las imágenes que intervinieron y `Cuadrante_referencia` dice en
  el marco de cuál están `Bbox` y las matrices del recorte (el del extremo noroeste).
  Una columna mayor que 2400 cae en el cuadrante de al lado.
- Si una de las imágenes no se puede leer, **sus píxeles entran como NaN** y salen
  del agregado. El registro se produce igualmente, parcial y marcado como tal:
  `Cuadrantes_faltantes` dice qué imágenes faltaron y `Fraccion_valida` qué parte
  del territorio llegó a medirse; en la matriz del recorte el hueco queda como
  `null`, no como cero. La corrida avisa por consola con el porcentaje de área
  perdida. **Si tu serie no admite registros parciales, fíltralos por
  `Fraccion_valida < 1` o por `Cuadrantes_faltantes` no vacío**; el pipeline no los
  descarta por su cuenta, porque para muchos análisis un municipio con el 98% de su
  territorio medido sigue sirviendo.

Un polígono que cruza el antimeridiano se rechaza: ese caso hay que partirlo en dos
antes de repartirlo.

### Municipios con islas, exclaves o enclaves

Un municipio tampoco tiene por qué ser un solo polígono. Puede tener islas o
exclaves —GeoJSON lo publica entonces como `MultiPolygon`— y puede tener huecos,
cuando otro municipio queda enclavado dentro de su territorio.

`extraer_geometria()` lee el límite completo, con sus partes y sus huecos, y es lo
que consume el cálculo de cobertura: una isla suma área aunque caiga en otro
cuadrante, y un enclave la resta. La cobertura se recorre parte por parte, porque
la envolvente de un municipio con una isla lejana incluye todo lo que hay en medio.

`extraer_coordenadas()` sigue existiendo para lo que solo necesita un contorno
—centroides, distancias, dibujos— pero **falla** ante un municipio multiparte o con
huecos en vez de devolver una de las partes como si fuera el municipio entero, que
es lo que hacía antes sin decirlo.

### Cuánto disco ocupa una corrida

Cada fecha en vuelo retiene sus gránulos —uno por cuadrante— desde la primera
descarga hasta que termina su último municipio, así que el pico de disco es:

```
pico ≈ NTL_MAX_FECHAS_CONCURRENTES × cuadrantes de la región × ~250 MB
```

Por omisión son **4 fechas a la vez**: con una región de dos cuadrantes, unos 2 GB.
Se ajusta con la variable de entorno `NTL_MAX_FECHAS_CONCURRENTES` o con
`run(..., max_concurrentes=N)`.

Es independiente de `chunks`, que decide cada cuántas fechas se guarda progreso.
Antes no había límite: se lanzaba una tarea por fecha y todas descargaban a la
vez, de modo que acotar el disco obligaba a pedir checkpoints que quizá no se
querían. Desde que un municipio puede necesitar cuatro cuadrantes en vez de uno,
el margen es cuatro veces menor.

### Cuándo falta una fila, y por qué

Una fila ausente no dice por qué está ausente. El pipeline distingue tres cosas
que antes eran indistinguibles:

| Situación | Qué pasa | Cómo se ve |
|---|---|---|
| Falta una de varias imágenes | Sale el registro, parcial | `Fraccion_valida < 1`, `Cuadrantes_faltantes` |
| No hay imagen, o la noche estaba nublada | No sale registro | Fila ausente. **Es un dato**, no un defecto |
| La cobertura y las imágenes se contradicen | `MedicionImposible` | Anotado en `sat.fallos` |
| Cualquier otro error | Propaga | Anotado en `sat.fallos` |

Antes las tres últimas se atrapaban y salían como «no hay datos»: en una serie de
diez años, una noche nublada y un defecto del código dejaban exactamente el mismo
hueco, y no había manera de decir cuál fue cuál.

Un fallo no detiene la corrida. Al terminar se resume y se guarda el detalle en
`data/fallos.parquet`:

```
⚠️ 3 de 6570 mediciones fallaron (0.05%): 2 KeyError, 1 OSError. Faltan esas
   filas de la serie y no es porque no hubiera imagen. El detalle está en `.fallos`.
```

```python
sat = SatelliteImagesAsync(municipios)
df = await sat.run(fechas)
if sat.fallos:                 # municipio, fecha, cuadrantes, tipo, mensaje
    print(pd.DataFrame(sat.fallos))
```

Con `run(..., estricto=True)` el primer fallo aborta, que es lo que quiere quien
reconstruye una serie desde cero. La API lo expone en el campo `fallos` de
`GET /jobs/{job_id}`: un job puede terminar `completed` y aun así traer filas de
menos.

---

## Ejemplos de uso desde código

En la carpeta `examples/` hay dos scripts listos para ejecutar desde la raíz del proyecto:

- `examples/sync_example.py`: uso básico de la versión síncrona (`SatelliteProcessor`).
- `examples/async_example.py`: uso básico de la versión asíncrona (`SatelliteImagesAsync`).

```bash
python examples/sync_example.py
python examples/async_example.py
```

Los dos ejemplos que consumían la API por HTTP se fueron con ella.

---

## Ejecución de tests

Instalación del entorno de pruebas (no hace falta `requirements.txt`, que
arrastra Jupyter y geopandas):

```bash
pip install -e ".[api,dev]"
```

```bash
python -m pytest              # toda la batería, menos las marcadas `lento`
python -m pytest -m lento     # geometría real a 2400x2400: ~2 minutos
python -m pytest tests/api    # solo la API
```

Cubren la lógica síncrona y asíncrona, la descarga, la API, la geometría de
cobertura y el reparto entre cuadrantes.

### Geometría real, y por qué está marcada aparte

Los 18 municipios de `ntl_data/` caben todos en un cuadrante y son polígonos
simples, así que **en producción el camino multicuadrante no se ejecuta nunca**:
estaba probado solo con figuras sintéticas y con municipios reales trasladados a
las esquinas de la retícula.

`tests/test_geometria_real.py` usa límites administrativos reales sin retocar
—Natural Earth, dominio público— de estados que sí cruzan: 21 de los 33 estados
de México cruzan una línea de 10 grados.

| Estado | Cuadrantes | Partes |
|---|---|---|
| Distrito Federal | 1 | 1 |
| Guanajuato | 3 | 1 |
| Quintana Roo | 2 | 5 (Cozumel, Isla Mujeres, Holbox) |
| Campeche | 4 | 2 |
| Sonora | 4 | 8 (islas del Golfo de California) |

Sonora es el caso que cruza los dos problemas difíciles: cuatro cuadrantes y
ocho partes, con islas en cuadrantes distintos del continente.

Intersecar exactamente uno de estos contra la retícula de 2400×2400 son decenas
de segundos, así que `pytest.ini` deselecciona las pesadas por omisión y CI las
corre siempre. Guanajuato se queda en la corrida normal para que un cruce real
se compruebe en cada cambio.

### Integración continua

`.github/workflows/pruebas.yml` corre la batería en cada push a `main` y
`develop` y en cada pull request, sobre Python 3.11 y 3.14. Un paso comprueba
que **no se salte ninguna prueba**: sin `pyproj`, las que comparan la cobertura
contra el área geodésica sobre WGS84 se saltaban en silencio, y son la
verdad-terreno de la tabla.

---

## Notas

- Se usa **Pydantic v2** (`model_dump`, `model_validate`) para validación y serialización de datos.
- Los resultados de mediciones se devuelven tipados como `MedicionResultado` en la API.
- Los archivos temporales y resultados intermedios se gestionan dentro del proyecto (por ejemplo, directorio `temp/`).

### Autores y coautores

- Proyecto desarrollado como trabajo terminal en ESCOM.
- Coautora: [Carolina Corral](https://github.com/carolinacorral).

Para dudas o contribuciones, abre un issue o un Pull Request en el repositorio.