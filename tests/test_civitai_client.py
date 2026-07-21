import unittest

from modules.civitai_client import CivitaiError, _choose_safe_file, _safe_filename


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


if __name__ == "__main__":
    unittest.main()
