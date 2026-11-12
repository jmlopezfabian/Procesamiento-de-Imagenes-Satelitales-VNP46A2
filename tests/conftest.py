"""
Shared pytest configuration and fixtures for los tres subpaquetes tests.
Adds project root to sys.path so modules can be imported when running pytest from repo root.
"""
import sys
from pathlib import Path

import pytest

# Project root (parent of tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Path to tests/fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_listado_json(fixtures_dir):
    """Listado JSON de LAADS (details/allData) con los cuadrantes h08v07 y h09v07 de VJ146A2."""
    import json
    return json.loads((fixtures_dir / "sample_listado.json").read_text(encoding="utf-8"))


@pytest.fixture
def municipios_json_path(fixtures_dir):
    """Path to sample municipios GeoJSON fixture."""
    return str(fixtures_dir / "municipios_sample.json")


@pytest.fixture
def sample_hdf5_path(tmp_path):
    """
    HDF5 mínimo con la forma de VNP46A2: radianza corregida y bandera de calidad.

    La bandera es obligatoria; un archivo sin ella se rechaza, así que el fixture
    la incluye con todos los píxeles en 0 (alta calidad).
    """
    import h5py
    import numpy as np

    h5_path = tmp_path / "sample.h5"
    grupo = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
    radiance = np.random.uniform(0, 100, size=(10, 10)).astype(np.float32)

    metadata_str = (
        "UpperLeftPointMtrs=(-1111950.519723,-555975.259861)\n"
        "LowerRightMtrs=(-1000755.467751,-666300.346481)\n"
    )

    with h5py.File(h5_path, "w") as f:

        f.attrs["ShortName"] = b"VNP46A2"
        grp = f.create_group(grupo)
        ds = grp.create_dataset("DNB_BRDF-Corrected_NTL", data=radiance)
        ds.attrs["_FillValue"] = np.float32(-999.9)
        ds.attrs["scale_factor"] = 1.0
        ds.attrs["units"] = b"nWatts/(cm^2 sr)"
        grp.create_dataset(
            "Mandatory_Quality_Flag", data=np.zeros((10, 10), dtype=np.uint8)
        )
        info = f.create_group("HDFEOS INFORMATION")
        info.create_dataset(
            "StructMetadata.0",
            data=np.array(metadata_str, dtype="S" + str(len(metadata_str) + 1)),
        )

    return str(h5_path)


@pytest.fixture
def mock_hdf5_file():
    """
    Factory that returns a mock h5py.File-like object with IMAGE_PATH and StructMetadata.
    Use when you need to avoid creating a real file (e.g. testing processor with custom matrix).
    """
    from unittest.mock import MagicMock
    import numpy as np

    def _make(image_matrix=None):
        if image_matrix is None:
            image_matrix = np.random.uniform(0, 50, size=(20, 20)).astype(np.float32)
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.__contains__ = MagicMock(return_value=True)
        mock_file.keys = MagicMock(return_value=["HDFEOS"])
        mock_file.__getitem__ = MagicMock(side_effect=lambda key: _dataset_for(key, mock_file, image_matrix))
        return mock_file

    def _dataset_for(key, root, image_matrix):
        grupo = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
        if key == f"{grupo}/Mandatory_Quality_Flag":
            mock_ds = MagicMock()
            calidad = np.zeros(image_matrix.shape, dtype=np.uint8)
            mock_ds.__getitem__ = MagicMock(return_value=calidad)
            return mock_ds
        if key == f"{grupo}/DNB_BRDF-Corrected_NTL":
            mock_ds = MagicMock()
            mock_ds.name = f"{grupo}/DNB_BRDF-Corrected_NTL"
            mock_ds.attrs = {"scale_factor": 1.0, "units": b"nWatts/(cm^2 sr)"}
            mock_ds.__call__ = MagicMock(return_value=image_matrix)
            mock_ds.__getitem__ = MagicMock(return_value=image_matrix)
            return mock_ds
        if key == "HDFEOS INFORMATION/StructMetadata.0":
            mock_ds = MagicMock()
            meta = (
                "UpperLeftPointMtrs=(-1111950.519723,-555975.259861)\n"
                "LowerRightMtrs=(-1000755.467751,-666300.346481)\n"
            )
            # numpy scalar so .tobytes().decode("utf-8") works as in utils.left_right_coords
            meta_val = np.array(meta, dtype="S" + str(len(meta) + 1)).flat[0]
            mock_ds.__call__ = MagicMock(return_value=meta_val)
            mock_ds.__getitem__ = MagicMock(return_value=meta_val)
            return mock_ds
        if key == "HDFEOS":
            mock_grp = MagicMock()
            mock_grp.keys = MagicMock(return_value=["GRIDS"])
            return mock_grp
        raise KeyError(key)

    return _make


@pytest.fixture(autouse=True)
def red_sin_esperas(monkeypatch):
    """
    Descargas sin esperas reales y con un limitador nuevo en cada prueba.

    Los reintentos duermen segundos de verdad y el bucket de peticiones va a
    2/s: con ellos cada prueba de red tardaría lo que tarda NASA. `esperas`
    recoge lo que el código habría dormido para poder comprobarlo.
    """
    import ntl.core.config as config
    import ntl.core.downloader as downloader
    from ntl.core.limitador import reiniciar_limitador

    esperas = []

    async def _dormir(segundos):
        esperas.append(segundos)

    monkeypatch.setattr(downloader, "_dormir", _dormir)
    monkeypatch.setattr(config, "MAX_PETICIONES_POR_SEG", 10_000.0)
    reiniciar_limitador()
    yield esperas
    reiniciar_limitador()


@pytest.fixture
def esperas(red_sin_esperas):
    """Lista de esperas (s) que los reintentos pidieron dormir."""
    return red_sin_esperas


@pytest.fixture(autouse=True)
def entorno_ntl_limpio(monkeypatch):
    """
    Pruebas independientes del entorno de quien las corre.

    `NTL_PRODUCTO` se lee al llamar (no al importar) y cambia el producto por
    omisión: con `NTL_PRODUCTO=VNP46A2` exportada (o en un .env, que config
    carga) fallaban 15 pruebas que esperan el VJ146A2 por omisión. Las pruebas
    que necesitan un valor lo fijan ellas con `monkeypatch.setenv`.
    """
    import os
    for nombre in [n for n in os.environ if n.startswith(("NTL_", "VNP46A1_"))]:
        monkeypatch.delenv(nombre, raising=False)
