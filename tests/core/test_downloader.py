"""Tests for ntl downloader with mocked HTTP and I/O."""
from unittest.mock import patch, MagicMock

import pytest

from ntl.core.downloader import find_file, download_file, is_valid_hdf5_file


class TestFindFile:
    def _resp(self, status=200, datos=None):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = datos
        return r

    def test_devuelve_el_downloads_link_del_cuadrante(self, sample_listado_json):
        with patch("ntl.core.downloader.requests.get",
                   return_value=self._resp(200, sample_listado_json)) as get:
            url = find_file(2024, 1, "h08v07")
        assert url.endswith("VJ146A2.A2024001.h08v07.002.2024003022532.h5")
        # Pide el endpoint JSON del producto por omisión.
        assert get.call_args.args[0].endswith("/api/v2/content/details/allData/5200/VJ146A2/2024/001/")

    def test_returns_none_when_status_not_200(self):
        with patch("ntl.core.downloader.requests.get", return_value=self._resp(404)):
            assert find_file(2024, 1, "h08v07") is None

    def test_returns_none_when_no_matching_link(self, sample_listado_json):
        with patch("ntl.core.downloader.requests.get",
                   return_value=self._resp(200, sample_listado_json)):
            assert find_file(2024, 1, "h99v99") is None

    def test_acepta_la_raiz_como_lista(self, sample_listado_json):
        with patch("ntl.core.downloader.requests.get",
                   return_value=self._resp(200, sample_listado_json["content"])):
            assert find_file(2024, 1, "h09v07").endswith("h09v07.002.2024003022532.h5")

    def test_no_pide_nada_si_la_fecha_esta_fuera_de_vigencia(self):
        with patch("ntl.core.downloader.requests.get") as get:
            assert find_file(2026, 307, "h08v07", "VNP46A2") is None
        get.assert_not_called()


class TestDownloadFile:
    def test_returns_path_when_download_ok_and_valid_h5(self, tmp_path):
        save_path = str(tmp_path / "downloaded.h5")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content = lambda chunk_size: [b"\x89HDF\r\n\x1a\n" + b"\x00" * 100]
        with patch("ntl.core.downloader.requests.get", return_value=mock_resp):
            with patch("ntl.core.downloader.is_valid_hdf5_file", return_value=True):
                result = download_file("http://example.com/file.h5", save_path)
        assert result == save_path

    def test_returns_none_when_response_not_200(self, tmp_path):
        save_path = str(tmp_path / "out.h5")
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("ntl.core.downloader.requests.get", return_value=mock_resp):
            result = download_file("http://example.com/file.h5", save_path)
        assert result is None

    def test_returns_none_when_valid_hdf5_fails_after_download(self, tmp_path):
        save_path = str(tmp_path / "out.h5")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content = lambda chunk_size: [b"not hdf5 content"]
        with patch("ntl.core.downloader.requests.get", return_value=mock_resp):
            with patch("ntl.core.downloader.is_valid_hdf5_file", return_value=False):
                with patch("ntl.core.downloader.os.remove"):
                    result = download_file("http://example.com/file.h5", save_path)
        assert result is None


class TestIsValidHdf5File:
    def test_returns_true_for_valid_hdf5(self, sample_hdf5_path):
        assert is_valid_hdf5_file(sample_hdf5_path) is True

    def test_returns_false_for_nonexistent_path(self):
        assert is_valid_hdf5_file("/nonexistent/path.h5") is False

    def test_returns_false_for_empty_file(self, tmp_path):
        empty = tmp_path / "empty.h5"
        empty.write_bytes(b"")
        assert is_valid_hdf5_file(str(empty)) is False

    def test_returns_false_for_html_content(self, tmp_path):
        fake = tmp_path / "fake.h5"
        fake.write_bytes(b"<!DOCTYPE html><html></html>")
        assert is_valid_hdf5_file(str(fake)) is False
