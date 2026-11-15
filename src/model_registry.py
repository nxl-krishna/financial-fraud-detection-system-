
import hashlib
import json
from pathlib import Path


def calculate_sha256(file_path):
    """Calculate the SHA-256 hash of a model artifact."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_active_model_registry(base_dir):
    """
    Load and validate the active model manifest.

    Returns:
        A tuple containing:
        - model metadata
        - resolved model artifact path
        - actual SHA-256 hash
        - resolved manifest path
    """

    base_dir = Path(base_dir).resolve()

    registry_dir = base_dir / "models" / "registry"

    pointer_path = registry_dir / "active_model.json"

    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"Active model pointer not found: {pointer_path}"
        )

    with open(
        pointer_path,
        "r",
        encoding="utf-8-sig",
    ) as file:
        pointer = json.load(file)

    if not isinstance(pointer, dict):
        raise RuntimeError(
            "Active model pointer must contain a JSON object."
        )

    manifest_filename = pointer.get("active_model_manifest")

    if (
        not isinstance(manifest_filename, str)
        or not manifest_filename
        or Path(manifest_filename).name != manifest_filename
        or not manifest_filename.endswith(".json")
    ):
        raise RuntimeError(
            "Invalid active model manifest filename."
        )

    manifest_path = (registry_dir / manifest_filename).resolve()

    try:
        manifest_path.relative_to(registry_dir.resolve())
    except ValueError:
        raise RuntimeError(
            "Model manifest must be inside the registry directory."
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Active model manifest not found: {manifest_path}"
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8-sig",
    ) as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise RuntimeError(
            "Model manifest must contain a JSON object."
        )

    if metadata.get("status") != "ACTIVE":
        raise RuntimeError(
            "Registered model is not ACTIVE."
        )

    artifact_relative_path = metadata.get("artifact")

    if (
        not isinstance(artifact_relative_path, str)
        or not artifact_relative_path
    ):
        raise RuntimeError(
            "Invalid model artifact path in registry."
        )

    artifact_path = (
        base_dir / artifact_relative_path
    ).resolve()

    try:
        artifact_path.relative_to(base_dir)
    except ValueError:
        raise RuntimeError(
            "Registered model artifact must be inside "
            "the project directory."
        )

    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Registered model artifact not found: {artifact_path}"
        )

    expected_hash = metadata.get("artifact_sha256")

    actual_hash = calculate_sha256(artifact_path)

    if (
        not isinstance(expected_hash, str)
        or actual_hash != expected_hash
    ):
        raise RuntimeError(
            "Model artifact SHA256 verification failed. "
            "The model file does not match the registered version."
        )

    return metadata, artifact_path, actual_hash, manifest_path
