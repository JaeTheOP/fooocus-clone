import unittest

from modules.civitai_client import (
    CivitaiError,
    _choose_safe_file,
    _safe_filename,
    extract_version_id,
)


class CivitaiClientTests(unittest.TestCase):
    def test_prefers_primary_safe_tensor(self):
        version = {
            "files": [
                {
                    "name": "unsafe.ckpt",
                    "primary": True,
                    "metadata": {"format": "PickleTensor"},
                    "virusScanResult": "Success",
                },
                {
                    "name": "safe.safetensors",
                    "primary": True,
                    "metadata": {"format": "SafeTensor"},
                    "virusScanResult": "Success",
                },
            ]
        }
        self.assertEqual(_choose_safe_file(version)["name"], "safe.safetensors")

    def test_rejects_failed_scan(self):
        version = {
            "files": [
                {
                    "name": "bad.safetensors",
                    "primary": True,
                    "metadata": {"format": "SafeTensor"},
                    "virusScanResult": "Danger",
                }
            ]
        }
        self.assertIsNone(_choose_safe_file(version))

    def test_sanitizes_filename(self):
        self.assertEqual(_safe_filename("../My Model!.safetensors"), "My Model_.safetensors")

    def test_rejects_non_safetensors(self):
        with self.assertRaises(CivitaiError):
            _safe_filename("model.ckpt")

    def test_extracts_numeric_version_id(self):
        self.assertEqual(extract_version_id("123456"), 123456)

    def test_extracts_version_id_from_model_url(self):
        self.assertEqual(
            extract_version_id("https://civitai.com/models/100/example?modelVersionId=200"),
            200,
        )

    def test_extracts_version_id_from_download_url(self):
        self.assertEqual(extract_version_id("https://civitai.com/api/download/models/300"), 300)

    def test_rejects_model_page_without_version(self):
        with self.assertRaises(CivitaiError):
            extract_version_id("https://civitai.com/models/100/example")


if __name__ == "__main__":
    unittest.main()
