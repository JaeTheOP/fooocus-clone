import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import colab_launcher


class ColabLauncherTests(unittest.TestCase):
    def test_split_references_accepts_commas_and_newlines(self):
        self.assertEqual(
            colab_launcher.split_references("123, 456\nhttps://civitai.com/api/download/models/789"),
            ["123", "456", "https://civitai.com/api/download/models/789"],
        )

    def test_configure_storage_creates_and_exports_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            child_env = {}
            with mock.patch.dict(os.environ, {"RF_DRIVE_ROOT": directory}, clear=False):
                colab_launcher.configure_storage(child_env)

            expected_keys = {
                "path_checkpoints",
                "path_loras",
                "path_vae",
                "path_embeddings",
                "path_outputs",
            }
            self.assertEqual(set(child_env), expected_keys)
            for value in child_env.values():
                self.assertTrue(Path(value).is_dir())
                self.assertFalse((Path(value) / ".renewed_fooocus_write_test").exists())

    def test_configure_storage_leaves_environment_unchanged_without_drive(self):
        child_env = {"existing": "value"}
        with mock.patch.dict(os.environ, {}, clear=True):
            colab_launcher.configure_storage(child_env)
        self.assertEqual(child_env, {"existing": "value"})

    def test_launch_fooocus_uses_selected_preset(self):
        child_env = {}
        completed = mock.Mock()
        with mock.patch.dict(os.environ, {"RF_PRESET": "anime"}, clear=False), mock.patch.object(
            colab_launcher, "run", return_value=completed
        ) as mocked_run:
            colab_launcher.launch_fooocus(child_env)

        command = mocked_run.call_args.args[0]
        self.assertEqual(command[:3], [str(colab_launcher.PYTHON), "launch.py", "--share"])
        self.assertIn("--always-high-vram", command)
        self.assertEqual(command[-2:], ["--preset", "anime"])
        self.assertEqual(child_env["GRADIO_ANALYTICS_ENABLED"], "False")


if __name__ == "__main__":
    unittest.main()
