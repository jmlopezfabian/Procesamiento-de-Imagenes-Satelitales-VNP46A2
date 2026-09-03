## Procesamiento de Imágenes Satelitales Black Marble

Proyecto para procesar imágenes satelitales de luminosidad nocturna del producto **VNP46A2**
(Black Marble, NASA) y obtener métricas de radianza por municipio.  
Incluye una **API asíncrona con FastAPI** para lanzar jobs de procesamiento en segundo plano y guardar resultados en Parquet.

### Componentes principales

El paquete `ntl/` está dividido por responsabilidad, no por modelo de concurrencia:

- `ntl/core/`: configuración, modelos, utilidades y descargas.
- `ntl/geometria/`: polígono municipal → cobertura por píxel. Corre **una vez por
  municipio**; su resultado es estático mientras no cambie la delimitación oficial.
- `ntl/radianza/`: cobertura → métricas de luminosidad. Corre **todos los días**
  sobre cada imagen, descargando en paralelo y agrupando por cuadrante.
- `api/`: aplicación FastAPI que expone el procesamiento como servicio HTTP.
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



## Uso de la API FastAPI (versión async)

Desde la raíz del proyecto:

```bash
# Opción 1: Con venv activado
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
uvicorn api.main:app --reload

# Opción 2: Sin activar venv (usa .venv automáticamente)
./run.sh
```

La documentación interactiva y la interfaz web estarán en:

- `http://localhost:8000` – Interfaz web para usar los endpoints
- `http://localhost:8000/docs` – Documentación Swagger

### Endpoints principales

- **`GET /municipios`**
  - Devuelve la lista de municipios disponibles para procesamiento.
  - Respuesta: `{ "municipios": ["iztapalapa", "coyoacan", ...] }`

- **`POST /jobs`**
  - Crea un job de procesamiento asíncrono.
  - Cuerpo (`JobRequest`):

    ```json
    {
      "municipios": ["iztapalapa"],
      "fecha_inicio": "2024-01-01",
      "fecha_fin": "2024-01-03",
      "chunks": 2
    }
    ```

  - Respuesta (`JobStatus`, HTTP 202):

    ```json
    {
      "job_id": "uuid-generado",
      "status": "pending",
      "progress": null,
      "created_at": "2024-01-01T00:00:00",
      "finished_at": null,
      "error": null,
      "total_results": 0
    }
    ```

- **`GET /jobs/{job_id}`**
  - Consulta el estado actual del job (`pending`, `running`, `completed`, `failed`).
  - Respuesta: `JobStatus`.

- **`GET /jobs/{job_id}/results`**
  - Devuelve los resultados del job una vez completado.
  - Respuesta (`JobResult`): contiene `results`, una lista de `MedicionResultado` serializados a JSON, por ejemplo:

    ```json
    {
      "job_id": "uuid-generado",
      "results": [
        {
          "Fecha": "2024-01-01",
          "Municipio": "iztapalapa",
          "Cantidad_de_pixeles": 114.39,
          "Suma_de_radianza": 1000.0,
          "Media_de_radianza": 10.0,
          "Desviacion_estandar_de_radianza": 1.0,
          "Maximo_de_radianza": 12.0,
          "Minimo_de_radianza": 8.0,
          "Percentil_25_de_radianza": 9.0,
          "Percentil_50_de_radianza": 10.0,
          "Percentil_75_de_radianza": 11.0
        }
      ]
    }
    ```

- **`DELETE /jobs/{job_id}`**
  - Cancela un job pendiente/en ejecución y lo elimina del store.

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

---

## Ejemplos de uso desde código

En la carpeta `examples/` hay tres scripts listos para ejecutar desde la raíz del proyecto:

- `examples/sync_example.py`: uso básico de la versión síncrona (`SatelliteProcessor`).
- `examples/async_example.py`: uso básico de la versión asíncrona (`SatelliteImagesAsync`).
- `examples/api_example.py`: consumo de la API FastAPI (requiere tener levantado `uvicorn api.main:app --reload`).

Ejemplos de ejecución:

```bash
python examples/sync_example.py
python examples/async_example.py
python examples/api_example.py
```

---

## Ejecución de tests

El proyecto usa `pytest` y tests para:
- Lógica síncrona y asíncrona.
- Descarga de archivos.
- API FastAPI (endpoints y manejo de jobs).

Desde la raíz del proyecto, con el entorno virtual activado:

```bash
python -m pytest          # Ejecuta toda la batería de tests
python -m pytest tests/api  # Solo tests de la API
```

---

## Notas

- Se usa **Pydantic v2** (`model_dump`, `model_validate`) para validación y serialización de datos.
- Los resultados de mediciones se devuelven tipados como `MedicionResultado` en la API.
- Los archivos temporales y resultados intermedios se gestionan dentro del proyecto (por ejemplo, directorio `temp/`).

### Autores y coautores

- Proyecto desarrollado como trabajo terminal en ESCOM.
- Coautora: [Carolina Corral](https://github.com/carolinacorral).

Para dudas o contribuciones, abre un issue o un Pull Request en el repositorio.