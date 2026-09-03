"""Integration tests for ntl SatelliteImagesAsync with mocked download/processing."""
import asyncio
import os
from datetime import date
from unittest.mock import patch, AsyncMock, MagicMock

import pandas as pd
import pytest

from ntl.radianza.lotes import SatelliteImagesAsync


@pytest.fixture
def mock_coord_data():
    """Fake CoordenadasPixeles-like object for init."""
    obj = MagicMock()
    obj.cuadrante = "h08v07"
    obj.pesos = [(1, 1, 1.0), (2, 1, 0.5), (2, 2, 1.0), (1, 2, 0.25)]
    return obj


@pytest.mark.asyncio
class TestSatelliteImagesAsyncRun:
    async def test_run_returns_dataframe_with_results(self, mock_coord_data):
        with patch("ntl.radianza.lotes.load_coord_data", return_value=mock_coord_data):
            sat = SatelliteImagesAsync("Iztapalapa")
        fake_results = [
            {
                "Fecha": date(2024, 1, 1),
                "Municipio": "iztapalapa",
                "Cantidad_de_pixeles": 4,
                "Suma_de_radianza": 10.0,
                "Media_de_radianza": 2.5,
                "Desviacion_estandar_de_radianza": 0.5,
                "Maximo_de_radianza": 3.0,
                "Minimo_de_radianza": 2.0,
                "Percentil_25_de_radianza": 2.0,
                "Percentil_50_de_radianza": 2.5,
                "Percentil_75_de_radianza": 3.0,
            }
        ]
        with patch.object(
            sat,
            "get_measures_for_date",
            new_callable=AsyncMock,
            return_value=fake_results,
        ):
            df = await sat.run(["01-01-24"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "Municipio" in df.columns
        assert "Media_de_radianza" in df.columns

    async def test_run_aggregates_multiple_dates(self, mock_coord_data):
        with patch("ntl.radianza.lotes.load_coord_data", return_value=mock_coord_data):
            sat = SatelliteImagesAsync("Iztapalapa")
        with patch.object(
            sat,
            "get_measures_for_date",
            new_callable=AsyncMock,
            side_effect=[
                [{"Fecha": date(2024, 1, 1), "Municipio": "iztapalapa", "Cantidad_de_pixeles": 1, "Suma_de_radianza": 1.0, "Media_de_radianza": 1.0, "Desviacion_estandar_de_radianza": 0.0, "Maximo_de_radianza": 1.0, "Minimo_de_radianza": 1.0, "Percentil_25_de_radianza": 1.0, "Percentil_50_de_radianza": 1.0, "Percentil_75_de_radianza": 1.0}],
                [{"Fecha": date(2024, 1, 2), "Municipio": "iztapalapa", "Cantidad_de_pixeles": 2, "Suma_de_radianza": 2.0, "Media_de_radianza": 1.0, "Desviacion_estandar_de_radianza": 0.0, "Maximo_de_radianza": 1.0, "Minimo_de_radianza": 1.0, "Percentil_25_de_radianza": 1.0, "Percentil_50_de_radianza": 1.0, "Percentil_75_de_radianza": 1.0}],
            ],
        ):
            df = await sat.run(["01-01-24", "02-01-24"], save_progress_enabled=False)
        assert len(df) == 2

    async def test_run_returns_empty_dataframe_when_no_results(self, mock_coord_data):
        with patch("ntl.radianza.lotes.load_coord_data", return_value=mock_coord_data):
            sat = SatelliteImagesAsync("Iztapalapa")
        with patch.object(
            sat,
            "get_measures_for_date",
            new_callable=AsyncMock,
            return_value=[],
        ):
            df = await sat.run(["01-01-24"], save_progress_enabled=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class _CoberturaFalsa:
    """Sustituto de CoordenadasPixeles con las piezas ya dadas."""

    def __init__(self, piezas):
        self._piezas = piezas

    @property
    def cuadrantes(self):
        return list(self._piezas)

    @property
    def piezas(self):
        return {c: p for c, p in self._piezas.items()}


@pytest.mark.asyncio
class TestDescargaDeVariosCuadrantes:
    """
    Un municipio puede necesitar varias imágenes y una imagen sirve a varios
    municipios. Lo que se comprueba aquí es la contabilidad: ni una descarga de
    más ni un archivo de cientos de megas que se quede en disco.
    """

    def _sat(self, coberturas):
        with patch("ntl.radianza.lotes.load_coord_data",
                   side_effect=lambda m, _: coberturas[m]):
            return SatelliteImagesAsync(list(coberturas))

    async def test_cada_cuadrante_se_descarga_una_vez_y_se_borra_al_final(self, tmp_path):
        """
        El cuadrante que dos municipios comparten se baja una vez, y no se borra
        hasta que el segundo terminó: borrarlo antes dejaría al vecino leyendo un
        archivo que ya no está.
        """
        coberturas = {
            # Uno repartido entre dos cuadrantes y otro que comparte uno de ellos.
            "repartido": _CoberturaFalsa({"h08v07": [(1, 1, 1.0)],
                                          "h09v07": [(0, 1, 0.5)]}),
            "vecino": _CoberturaFalsa({"h09v07": [(3, 3, 1.0)]}),
        }
        sat = self._sat(coberturas)

        descargas = []

        async def descarga(session, url, save_path):
            descargas.append(url)
            open(save_path, "w").write("h5")
            return save_path

        existentes_al_procesar = []

        def procesar(rutas, piezas, date_obj, municipio, delete_files=True):
            existentes_al_procesar.append(
                {c: os.path.exists(r) for c, r in rutas.items()}
            )
            return None

        with patch("ntl.radianza.lotes.TEMP_DIR", tmp_path), \
             patch("ntl.core.config.TEMP_DIR", tmp_path), \
             patch("ntl.radianza.lotes.temp_path", side_effect=lambda n: tmp_path / n), \
             patch("ntl.radianza.lotes.find_file_async",
                   new_callable=AsyncMock, side_effect=lambda s, y, d, c: f"http://x/{c}.h5"), \
             patch("ntl.radianza.lotes.download_file_async", side_effect=descarga), \
             patch("ntl.radianza.lotes.process_image_mosaico", side_effect=procesar):
            await sat.get_measures_for_date(None, "01-01-24")

        assert descargas == ["http://x/h08v07.h5", "http://x/h09v07.h5"], \
            "el cuadrante compartido se descargó más de una vez"
        # Cada municipio vio sus archivos en disco cuando le tocó procesar.
        assert all(all(presentes.values()) for presentes in existentes_al_procesar)
        assert len(existentes_al_procesar) == 2
        # Y al final no queda ninguno ocupando disco.
        assert list(tmp_path.glob("*.h5")) == []
        assert sat.cache_h5_files == {}

    async def test_al_municipio_repartido_le_llegan_sus_dos_imagenes(self):
        coberturas = {
            "repartido": _CoberturaFalsa({"h08v07": [(1, 1, 1.0)],
                                          "h09v07": [(0, 1, 0.5)]}),
        }
        sat = self._sat(coberturas)

        async def descarga(session, year, day, cuadrante, date_obj):
            return f"/tmp/{cuadrante}.h5"

        with patch.object(sat, "_download_and_cache_h5", side_effect=descarga), \
             patch.object(sat, "_borrar_cuadrante"), \
             patch("ntl.radianza.lotes.process_image_mosaico", return_value=None) as proc:
            await sat.get_measures_for_date(None, "01-01-24")

        rutas = proc.call_args.args[0]
        assert rutas == {"h08v07": "/tmp/h08v07.h5", "h09v07": "/tmp/h09v07.h5"}


@pytest.mark.asyncio
class TestLimiteDeFechasConcurrentes:
    """
    Antes se lanzaba una tarea por fecha y todas descargaban a la vez: un año de
    una región de dos cuadrantes son 730 gránulos simultáneos en disco. El
    semáforo va por fecha —no por descarga— porque una fecha retiene sus
    archivos hasta que termina su último municipio, así que limitar las descargas
    no limitaría cuántos hay residentes.
    """

    def _sat(self, coberturas):
        with patch("ntl.radianza.lotes.load_coord_data",
                   side_effect=lambda m, _: coberturas[m]):
            return SatelliteImagesAsync(list(coberturas))

    def _espia_de_concurrencia(self):
        """Cuenta cuántas fechas hay en vuelo y se queda con el máximo."""
        estado = {"en_vuelo": 0, "pico": 0, "fechas": []}

        async def medir(session, fecha, estricto=False):
            estado["en_vuelo"] += 1
            estado["pico"] = max(estado["pico"], estado["en_vuelo"])
            estado["fechas"].append(fecha)
            # Cede el control: sin esto las corrutinas no se solaparían nunca y
            # la prueba pasaría aunque el semáforo no existiera.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            estado["en_vuelo"] -= 1
            return [{"Fecha": fecha, "Municipio": "iztapalapa"}]

        return estado, medir

    @pytest.mark.parametrize("limite", [1, 2, 4])
    async def test_nunca_hay_mas_fechas_en_vuelo_que_el_limite(self, mock_coord_data, limite):
        with patch("ntl.radianza.lotes.load_coord_data", return_value=mock_coord_data):
            sat = SatelliteImagesAsync("Iztapalapa")
        fechas = [f"{d:02d}-01-24" for d in range(1, 13)]
        estado, medir = self._espia_de_concurrencia()

        with patch.object(sat, "get_measures_for_date", side_effect=medir):
            df = await sat.run(fechas, save_progress_enabled=False,
                               max_concurrentes=limite)

        assert estado["pico"] <= limite, (
            f"llegaron a correr {estado['pico']} fechas a la vez con límite {limite}")
        assert len(df) == len(fechas), "acotar la concurrencia perdió fechas"

    async def test_el_pico_de_archivos_en_disco_queda_acotado(self, tmp_path):
        """
        La aserción que de verdad importa: no cuántas corrutinas hay, sino
        cuántos gránulos de cientos de megas coexisten en el volumen.
        """
        coberturas = {
            "repartido": _CoberturaFalsa({"h08v07": [(1, 1, 1.0)],
                                          "h09v07": [(0, 1, 0.5)]}),
        }
        sat = self._sat(coberturas)
        fechas = [f"{d:02d}-01-24" for d in range(1, 11)]
        pico = {"h5": 0}

        async def descarga(session, url, save_path):
            open(save_path, "w").write("h5")
            pico["h5"] = max(pico["h5"], len(list(tmp_path.glob("*.h5"))))
            await asyncio.sleep(0)
            return save_path

        with patch("ntl.radianza.lotes.TEMP_DIR", tmp_path), \
             patch("ntl.core.config.TEMP_DIR", tmp_path), \
             patch("ntl.radianza.lotes.temp_path", side_effect=lambda n: tmp_path / n), \
             patch("ntl.radianza.lotes.find_file_async",
                   new_callable=AsyncMock, side_effect=lambda s, y, d, c: f"http://x/{c}.h5"), \
             patch("ntl.radianza.lotes.download_file_async", side_effect=descarga), \
             patch("ntl.radianza.lotes.process_image_mosaico", return_value=None), \
             patch("ntl.radianza.lotes.cleanup_temp_files"):
            await sat.run(fechas, save_progress_enabled=False, max_concurrentes=2)

        # 2 fechas en vuelo x 2 cuadrantes cada una.
        assert pico["h5"] <= 4, f"llegó a haber {pico['h5']} gránulos en disco"
        assert list(tmp_path.glob("*.h5")) == [], "quedaron archivos sin borrar"

    async def test_los_resultados_no_dependen_del_limite(self, mock_coord_data):
        """Acotar el disco no puede cambiar los datos."""
        fechas = [f"{d:02d}-01-24" for d in range(1, 8)]
        obtenidos = []
        for limite in (1, 3, len(fechas)):
            with patch("ntl.radianza.lotes.load_coord_data", return_value=mock_coord_data):
                sat = SatelliteImagesAsync("Iztapalapa")
            _, medir = self._espia_de_concurrencia()
            with patch.object(sat, "get_measures_for_date", side_effect=medir):
                df = await sat.run(fechas, save_progress_enabled=False,
                                   max_concurrentes=limite)
            obtenidos.append(sorted(df["Fecha"].tolist()))
        assert obtenidos[0] == obtenidos[1] == obtenidos[2] == sorted(fechas)

    async def test_una_fecha_que_revienta_no_se_queda_con_el_permiso(self, mock_coord_data):
        """
        Con `acquire`/`release` en vez de `async with`, una excepción dejaría el
        permiso tomado y la corrida se iría bloqueando hasta pararse.
        """
        with patch("ntl.radianza.lotes.load_coord_data", return_value=mock_coord_data):
            sat = SatelliteImagesAsync("Iztapalapa")
        fechas = [f"{d:02d}-01-24" for d in range(1, 7)]

        async def medir(session, fecha, estricto=False):
            await asyncio.sleep(0)
            raise RuntimeError("boom")

        with patch.object(sat, "get_measures_for_date", side_effect=medir):
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(
                    sat.run(fechas, save_progress_enabled=False, max_concurrentes=1),
                    timeout=5,
                )
        # El semáforo quedó libre: si no, el wait_for habría expirado.

    async def test_el_limite_tambien_aplica_procesando_por_chunks(self, mock_coord_data):
        """
        `chunks` acotaba la concurrencia de rebote, pero existe para guardar
        progreso. El límite tiene que valer también ahí.
        """
        with patch("ntl.radianza.lotes.load_coord_data", return_value=mock_coord_data):
            sat = SatelliteImagesAsync("Iztapalapa")
        fechas = [f"{d:02d}-01-24" for d in range(1, 13)]
        estado, medir = self._espia_de_concurrencia()

        with patch.object(sat, "get_measures_for_date", side_effect=medir):
            df = await sat.run(fechas, chunks=6, save_progress_enabled=False,
                               max_concurrentes=2)

        assert estado["pico"] <= 2
        assert len(df) == len(fechas)

    async def test_por_omision_toma_el_valor_de_configuracion(self, mock_coord_data):
        with patch("ntl.radianza.lotes.load_coord_data", return_value=mock_coord_data):
            sat = SatelliteImagesAsync("Iztapalapa")
        fechas = [f"{d:02d}-01-24" for d in range(1, 13)]
        estado, medir = self._espia_de_concurrencia()

        with patch("ntl.radianza.lotes.MAX_FECHAS_CONCURRENTES", 3), \
             patch.object(sat, "get_measures_for_date", side_effect=medir):
            await sat.run(fechas, save_progress_enabled=False)

        assert estado["pico"] <= 3


@pytest.mark.asyncio
class TestUnFalloNoDesapareceEnSilencio:
    """
    Una fila que no está no dice por qué no está. Antes «esa noche estaba
    nublado» y «el código se rompió» dejaban exactamente el mismo hueco, y a los
    dos años no había manera de decir cuál fue cuál.
    """

    def _sat(self, coberturas):
        with patch("ntl.radianza.lotes.load_coord_data",
                   side_effect=lambda m, _: coberturas[m]):
            return SatelliteImagesAsync(list(coberturas))

    def _coberturas(self):
        return {
            "bueno": _CoberturaFalsa({"h08v07": [(1, 1, 1.0)]}),
            "malo": _CoberturaFalsa({"h08v07": [(2, 2, 1.0)]}),
        }

    async def _correr(self, sat, procesar, **kw):
        async def descarga(session, year, day, cuadrante, date_obj):
            return f"/tmp/{cuadrante}.h5"

        with patch.object(sat, "_download_and_cache_h5", side_effect=descarga), \
             patch.object(sat, "_borrar_cuadrante"), \
             patch("ntl.radianza.lotes.process_image_mosaico", side_effect=procesar):
            return await sat.get_measures_for_date(None, "01-01-24", **kw)

    def _procesar(self, rutas, piezas, date_obj, municipio, delete_files=True):
        if municipio == "malo":
            raise KeyError("DNB_BRDF-Corrected_NTL")
        return MagicMock(model_dump=lambda: {"Municipio": municipio})

    async def test_un_municipio_que_revienta_no_tumba_a_los_demas(self):
        sat = self._sat(self._coberturas())
        resultados = await self._correr(sat, self._procesar)
        assert [r["Municipio"] for r in resultados] == ["bueno"]

    async def test_el_fallo_queda_registrado_con_su_contexto(self):
        sat = self._sat(self._coberturas())
        await self._correr(sat, self._procesar)

        assert len(sat.fallos) == 1
        fallo = sat.fallos[0]
        assert fallo["municipio"] == "malo"
        assert fallo["fecha"] == "2024-01-01"
        assert fallo["cuadrantes"] == ["h08v07"]
        assert fallo["tipo"] == "KeyError"
        assert "DNB_BRDF-Corrected_NTL" in fallo["mensaje"]

    async def test_sin_medicion_no_es_un_fallo(self):
        """
        Una noche nublada devuelve None y deja la fila fuera: eso es un dato,
        no un defecto, y no debe ensuciar la lista de fallos.
        """
        sat = self._sat(self._coberturas())
        resultados = await self._correr(
            sat, lambda *a, **k: None)
        assert resultados == []
        assert sat.fallos == []

    async def test_estricto_aborta_en_el_primer_fallo(self):
        sat = self._sat(self._coberturas())
        with pytest.raises(KeyError):
            await self._correr(sat, self._procesar, estricto=True)
        assert len(sat.fallos) == 1

    async def test_el_fallo_no_rompe_el_contador_de_referencias(self, tmp_path):
        """
        El decremento va fuera del try a propósito: si un municipio revienta, su
        cuadrante tiene que liberarse igual o el archivo se queda en disco para
        siempre. Queda fijado aquí antes de que alguien lo mueva dentro.
        """
        coberturas = self._coberturas()
        sat = self._sat(coberturas)

        async def descarga(session, url, save_path):
            open(save_path, "w").write("h5")
            return save_path

        with patch("ntl.radianza.lotes.TEMP_DIR", tmp_path), \
             patch("ntl.core.config.TEMP_DIR", tmp_path), \
             patch("ntl.radianza.lotes.temp_path", side_effect=lambda n: tmp_path / n), \
             patch("ntl.radianza.lotes.find_file_async",
                   new_callable=AsyncMock, side_effect=lambda s, y, d, c: f"http://x/{c}.h5"), \
             patch("ntl.radianza.lotes.download_file_async", side_effect=descarga), \
             patch("ntl.radianza.lotes.process_image_mosaico",
                   side_effect=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))):
            await sat.get_measures_for_date(None, "01-01-24")

        assert len(sat.fallos) == 2
        assert list(tmp_path.glob("*.h5")) == [], \
            "el cuadrante no se liberó cuando el municipio falló"
        assert sat.cache_h5_files == {}

    async def test_el_resumen_final_cuenta_los_fallos(self, mock_coord_data, capsys, tmp_path):
        with patch("ntl.radianza.lotes.load_coord_data", return_value=mock_coord_data):
            sat = SatelliteImagesAsync("Iztapalapa")

        async def medir(session, fecha, estricto=False):
            if fecha == "02-01-24":
                sat.fallos.append({"municipio": "iztapalapa", "fecha": fecha,
                                   "cuadrantes": ["h08v07"], "tipo": "OSError",
                                   "mensaje": "disco lleno"})
                return []
            return [{"Fecha": fecha, "Municipio": "iztapalapa"}]

        with patch.object(sat, "get_measures_for_date", side_effect=medir):
            await sat.run(["01-01-24", "02-01-24"], save_progress_enabled=False)

        salida = capsys.readouterr().out
        assert "1 de 2 mediciones fallaron" in salida
        assert "1 OSError" in salida

    async def test_sin_fallos_no_imprime_nada(self, mock_coord_data, capsys):
        with patch("ntl.radianza.lotes.load_coord_data", return_value=mock_coord_data):
            sat = SatelliteImagesAsync("Iztapalapa")

        async def medir(session, fecha, estricto=False):
            return [{"Fecha": fecha, "Municipio": "iztapalapa"}]

        with patch.object(sat, "get_measures_for_date", side_effect=medir):
            await sat.run(["01-01-24"], save_progress_enabled=False)

        assert "mediciones fallaron" not in capsys.readouterr().out

    async def test_el_detalle_se_persiste_al_lado_de_los_resultados(
            self, mock_coord_data, tmp_path):
        """Un resumen impreso en una terminal que se cerró no es evidencia."""
        with patch("ntl.radianza.lotes.load_coord_data", return_value=mock_coord_data):
            sat = SatelliteImagesAsync("Iztapalapa")

        async def medir(session, fecha, estricto=False):
            sat.fallos.append({"municipio": "iztapalapa", "fecha": fecha,
                               "cuadrantes": ["h08v07"], "tipo": "KeyError",
                               "mensaje": "falta el dataset"})
            return []

        with patch("ntl.radianza.lotes.data_path",
                   side_effect=lambda n: tmp_path / n), \
             patch.object(sat, "get_measures_for_date", side_effect=medir):
            await sat.run(["01-01-24"], save_progress_enabled=True)

        guardado = pd.read_parquet(tmp_path / "fallos.parquet")
        assert len(guardado) == 1
        assert guardado.iloc[0]["municipio"] == "iztapalapa"
        assert guardado.iloc[0]["tipo"] == "KeyError"
