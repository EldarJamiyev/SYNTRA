"""Offline runtime matrix for the raw native-model benchmark."""

from __future__ import annotations

from typing import Any


ENVIRONMENT_SPECS: dict[str, dict[str, str]] = {
    "python": {
        "image": "python:3.12-slim",
        "test_extension": ".py",
        "runtime": "python",
    },
    "typescript": {
        "image": "node:20-slim",
        "test_extension": ".ts",
        "runtime": "node",
    },
    "go": {
        "image": "golang:1.22-alpine",
        "test_extension": ".go",
        "runtime": "go",
    },
    "rust": {
        "image": "rust:1.78-slim",
        "test_extension": ".rs",
        "runtime": "cargo",
    },
}

SUPPORTED_LANGUAGES = frozenset(ENVIRONMENT_SPECS)


def resolve_environment(language: str, image_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Return a validated language runtime and its selected local image tag."""
    if language not in ENVIRONMENT_SPECS:
        raise ValueError("Unsupported benchmark language: " + str(language))
    spec = dict(ENVIRONMENT_SPECS[language])
    overrides = image_overrides or {}
    image = overrides.get(language)
    if image is None:
        image = spec["image"]
    if not isinstance(image, str) or not image.strip() or len(image) > 256:
        raise ValueError("A Docker image override must be a non-empty bounded string.")
    spec["language"] = language
    spec["image"] = image.strip()
    return spec


def expected_test_filename(language: str, phase: str) -> str:
    """Return the only permitted hidden-test filename for a language and phase."""
    if phase not in {"target", "regression"}:
        raise ValueError("Unknown benchmark test phase.")
    spec = ENVIRONMENT_SPECS.get(language)
    if spec is None:
        raise ValueError("Unsupported benchmark language: " + str(language))
    return "test_" + phase + spec["test_extension"]
