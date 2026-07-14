from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from service.model_registry import ModelRegistry


def _catalog(model_id: str, weights_name: str = "model.pt") -> str:
    return json.dumps(
        {
            "models": [
                {
                    "id": model_id,
                    "name": model_id,
                    "sensorTypes": ["wide", "unknown"],
                    "defaultFor": ["wide", "unknown"],
                    "weightsPath": weights_name,
                    "enabled": True,
                }
            ]
        }
    )


class ModelRegistryResilienceTests(unittest.TestCase):
    def test_legacy_pyronear_id_resolves_to_canonical_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.pt").write_bytes(b"weights")
            catalog_path = root / "catalog.json"
            catalog_path.write_text(_catalog("pyrone-yolov8s-wide"), encoding="utf-8")
            registry = ModelRegistry(
                SimpleNamespace(model_catalog_path=catalog_path, repo_dir=root)
            )

            self.assertEqual(
                registry.resolve("pyronear-yolov8s-wide", "wide"),
                "pyrone-yolov8s-wide",
            )
            model = registry.get("pyronear-yolov8s-wide")
            self.assertIsNotNone(model)
            self.assertEqual(model["id"], "pyrone-yolov8s-wide")
            self.assertTrue(model["weightsPresent"])

    def test_catalog_change_is_reloaded_without_restarting_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(_catalog("first-model"), encoding="utf-8")
            registry = ModelRegistry(
                SimpleNamespace(model_catalog_path=catalog_path, repo_dir=root)
            )
            initial_revision = registry.revision

            catalog_path.write_text(_catalog("second-model", "different-name.pt"), encoding="utf-8")

            self.assertIsNone(registry.get("first-model"))
            self.assertEqual(registry.get("second-model")["id"], "second-model")
            self.assertGreater(registry.revision, initial_revision)

    def test_transient_catalog_read_failure_retries_same_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(_catalog("first-model"), encoding="utf-8")
            registry = ModelRegistry(
                SimpleNamespace(model_catalog_path=catalog_path, repo_dir=root)
            )
            replacement = _catalog("replacement-model", "replacement-weights.pt")
            catalog_path.write_text(replacement, encoding="utf-8")
            original_read_text = Path.read_text
            failed_once = False

            def flaky_read_text(path: Path, *args, **kwargs):
                nonlocal failed_once
                if path == catalog_path and not failed_once:
                    failed_once = True
                    raise OSError("temporary read failure")
                return original_read_text(path, *args, **kwargs)

            with patch.object(Path, "read_text", autospec=True, side_effect=flaky_read_text):
                self.assertEqual(registry.get("first-model")["id"], "first-model")
                self.assertIn("temporary read failure", registry.last_reload_error)
                self.assertEqual(
                    registry.get("replacement-model")["id"],
                    "replacement-model",
                )

            self.assertEqual(registry.last_reload_error, "")

    def test_temporarily_missing_catalog_preserves_last_valid_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(_catalog("first-model"), encoding="utf-8")
            registry = ModelRegistry(
                SimpleNamespace(model_catalog_path=catalog_path, repo_dir=root)
            )
            initial_revision = registry.revision

            catalog_path.unlink()

            self.assertEqual(registry.get("first-model")["id"], "first-model")
            self.assertIsNone(registry.get("smoke-wide-v1"))
            self.assertIn("model catalog is unavailable", registry.last_reload_error)
            self.assertEqual(registry.revision, initial_revision)

            catalog_path.write_text(
                _catalog("replacement-model", "replacement-weights.pt"),
                encoding="utf-8",
            )

            self.assertIsNone(registry.get("first-model"))
            self.assertEqual(
                registry.get("replacement-model")["id"],
                "replacement-model",
            )
            self.assertEqual(registry.last_reload_error, "")

    def test_malformed_catalog_preserves_last_valid_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(_catalog("first-model"), encoding="utf-8")
            registry = ModelRegistry(
                SimpleNamespace(model_catalog_path=catalog_path, repo_dir=root)
            )
            initial_revision = registry.revision

            catalog_path.write_text("{not-json", encoding="utf-8")

            self.assertEqual(registry.get("first-model")["id"], "first-model")
            self.assertIsNone(registry.get("smoke-wide-v1"))
            self.assertNotEqual(registry.last_reload_error, "")
            self.assertEqual(registry.revision, initial_revision)

    def test_missing_catalog_uses_defaults_only_on_initial_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "catalog.json"
            registry = ModelRegistry(
                SimpleNamespace(model_catalog_path=catalog_path, repo_dir=root)
            )

            self.assertEqual(
                registry.default_model_id("wide"),
                "smoke-wide-v1",
            )
            self.assertEqual(registry.last_reload_error, "")

    def test_weights_presence_is_rechecked_without_catalog_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(_catalog("late-weights"), encoding="utf-8")
            registry = ModelRegistry(
                SimpleNamespace(model_catalog_path=catalog_path, repo_dir=root)
            )

            self.assertFalse(registry.get("late-weights")["weightsPresent"])
            (root / "model.pt").write_bytes(b"weights")
            self.assertTrue(registry.get("late-weights")["weightsPresent"])


if __name__ == "__main__":
    unittest.main()
