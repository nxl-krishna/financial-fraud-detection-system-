
import hashlib
import json

import pytest

from src.model_registry import load_active_model_registry


@pytest.fixture
def registry_environment(tmp_path):
    """Create an isolated temporary model registry."""

    registry_dir = tmp_path / "models" / "registry"
    registry_dir.mkdir(parents=True)

    artifact_dir = tmp_path / "models"
    artifact_dir.mkdir(exist_ok=True)

    artifact_path = artifact_dir / "test_model.joblib"
    artifact_path.write_bytes(b"test model artifact")

    artifact_hash = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()

    manifest = {
        "model_id": "test-fraud-model",
        "model_version": "1.0.0",
        "model_name": "Test Random Forest",
        "algorithm": "RandomForestClassifier",
        "artifact": "models/test_model.joblib",
        "artifact_sha256": artifact_hash,
        "feature_count": 37,
        "decision_threshold": 0.5,
        "status": "ACTIVE",
    }

    manifest_path = registry_dir / "model_v1.0.0.json"
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    pointer_path = registry_dir / "active_model.json"
    pointer_path.write_text(
        json.dumps({
            "active_model_manifest": "model_v1.0.0.json"
        }),
        encoding="utf-8",
    )

    return {
        "base_dir": tmp_path,
        "registry_dir": registry_dir,
        "artifact_path": artifact_path,
        "manifest_path": manifest_path,
        "pointer_path": pointer_path,
        "manifest": manifest,
        "artifact_hash": artifact_hash,
    }


def test_load_active_model_registry_success(registry_environment):
    env = registry_environment

    metadata, artifact, actual_hash, manifest_path = (
        load_active_model_registry(env["base_dir"])
    )

    assert metadata["model_id"] == "test-fraud-model"
    assert metadata["model_version"] == "1.0.0"
    assert metadata["status"] == "ACTIVE"

    assert artifact == env["artifact_path"]
    assert actual_hash == env["artifact_hash"]
    assert manifest_path == env["manifest_path"]


def test_missing_active_pointer_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Active model pointer"):
        load_active_model_registry(tmp_path)


def test_invalid_manifest_filename_is_rejected(registry_environment):
    env = registry_environment

    env["pointer_path"].write_text(
        json.dumps({
            "active_model_manifest": "../outside.json"
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Invalid active model manifest"):
        load_active_model_registry(env["base_dir"])


def test_missing_manifest_raises_error(registry_environment):
    env = registry_environment

    env["pointer_path"].write_text(
        json.dumps({
            "active_model_manifest": "missing_model.json"
        }),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Active model manifest"):
        load_active_model_registry(env["base_dir"])


def test_inactive_model_is_rejected(registry_environment):
    env = registry_environment

    manifest = env["manifest"].copy()
    manifest["status"] = "INACTIVE"

    env["manifest_path"].write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not ACTIVE"):
        load_active_model_registry(env["base_dir"])


def test_artifact_outside_project_is_rejected(registry_environment):
    env = registry_environment

    manifest = env["manifest"].copy()
    manifest["artifact"] = "../outside_model.joblib"

    env["manifest_path"].write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="inside the project directory"):
        load_active_model_registry(env["base_dir"])


def test_modified_artifact_fails_hash_verification(registry_environment):
    env = registry_environment

    env["artifact_path"].write_bytes(
        b"modified model artifact"
    )

    with pytest.raises(RuntimeError, match="SHA256 verification failed"):
        load_active_model_registry(env["base_dir"])
