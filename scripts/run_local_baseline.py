"""Read-only local readiness gate for SYNTRA's native baseline workflow.

This command performs actual local PyTorch device operations and Docker image
inspection. It never pulls, builds, tags, or runs containers and never accesses
a model registry or hosted inference service.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_epochs

REQUIRED_DOCKER_IMAGES = (
    "python:3.12-slim",
    "node:20-slim",
    "golang:1.22-alpine",
    "rust:1.78-slim",
)
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPO_DIGEST = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_MIN_PYTHON = (3, 12)


class PreflightError(RuntimeError):
    """A configuration or readiness check could not be completed safely."""


def _load_profile(config: str | Path) -> dict[str, Any]:
    try:
        path = evaluate_epochs._resolve_profile_path(Path(config))
        profile = evaluate_epochs.load_profile(path)
        evaluate_epochs._validate_profile_values(profile)
        return profile
    except (OSError, ValueError, evaluate_epochs.OrchestrationError) as exc:
        raise PreflightError("Training profile is not valid: " + str(exc)) from None


def _device_status(torch: Any, device_name: str) -> dict[str, Any]:
    if device_name == "cpu":
        available = True
    elif device_name == "cuda":
        available = bool(torch.cuda.is_available())
    elif device_name == "mps":
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        available = bool(backend and backend.is_available())
    else:
        raise ValueError("Unsupported device name: " + device_name)

    result: dict[str, Any] = {
        "available": available,
        "ready": False,
        "device": device_name,
        "detail": "backend unavailable in this local PyTorch installation",
    }
    if not available:
        return result

    try:
        device = torch.device(device_name)
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device)
        right = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32, device=device)
        product = left @ right
        if device_name == "cuda":
            torch.cuda.synchronize(device)
            result["name"] = torch.cuda.get_device_name(device)
            properties = torch.cuda.get_device_properties(device)
            result["total_memory_bytes"] = int(properties.total_memory)
        elif device_name == "mps":
            torch.mps.synchronize()
            result["name"] = "Apple Metal Performance Shaders"
        observed = product.detach().to("cpu")
        expected = torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.float32)
        if not torch.equal(observed, expected):
            raise RuntimeError("device matrix operation returned an unexpected result")
        result["ready"] = True
        result["detail"] = "local tensor allocation, matrix operation, synchronization, and result verification passed"
    except Exception as exc:
        result["detail"] = type(exc).__name__ + ": " + str(exc)
    return result


def probe_torch_devices(preference: str = "auto") -> dict[str, Any]:
    if preference not in {"auto", "cpu", "cuda", "mps"}:
        raise PreflightError("Training device must be auto, cpu, cuda, or mps.")
    python_ready = sys.version_info >= _MIN_PYTHON
    report: dict[str, Any] = {
        "python": {
            "version": sys.version.split()[0],
            "minimum": "3.12",
            "ready": python_ready,
        },
        "pytorch": {"ready": False, "version": None, "compiled_cuda": None},
        "devices": {},
        "requested_device": preference,
        "selected_device": None,
        "selected_device_ready": False,
    }
    try:
        import torch
    except Exception as exc:
        report["pytorch"]["error"] = type(exc).__name__ + ": " + str(exc)
        report["devices"] = {
            name: {"available": False, "ready": False, "detail": "PyTorch could not be imported"}
            for name in ("cpu", "cuda", "mps")
        }
        return report

    report["pytorch"].update({
        "ready": True,
        "version": str(torch.__version__),
        "compiled_cuda": getattr(torch.version, "cuda", None),
        "mps_built": bool(
            getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_built()
        ),
    })
    devices = {name: _device_status(torch, name) for name in ("cpu", "cuda", "mps")}
    report["devices"] = devices
    if preference == "auto":
        selected = (
            "cuda" if devices["cuda"]["available"]
            else "mps" if devices["mps"]["available"]
            else "cpu"
        )
    else:
        selected = preference
    report["selected_device"] = selected
    report["selected_device_ready"] = bool(devices[selected]["ready"])
    report["cpu_required_ready"] = bool(devices["cpu"]["ready"])
    return report


def _expected_tag_is_present(image: str, tags: Any) -> bool:
    if not isinstance(tags, list):
        return False
    accepted = {
        image.casefold(),
        ("docker.io/library/" + image).casefold(),
        ("index.docker.io/library/" + image).casefold(),
    }
    return any(isinstance(tag, str) and tag.casefold() in accepted for tag in tags)


def validate_docker_image_inspect(image: str, payload: Any) -> dict[str, Any]:
    """Validate the local daemon's inspect record; this function makes no calls."""
    summary: dict[str, Any] = {
        "image": image,
        "present": False,
        "ready": False,
        "id_sha256_valid": False,
        "repository_digest_valid": False,
        "tag_matches": False,
        "os": None,
        "architecture": None,
        "error": None,
    }
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        summary["error"] = "docker image inspect did not return exactly one image object"
        return summary
    item = payload[0]
    summary["present"] = True
    image_id = item.get("Id")
    summary["id_sha256_valid"] = isinstance(image_id, str) and bool(_SHA256.fullmatch(image_id))
    tags = item.get("RepoTags")
    summary["tag_matches"] = _expected_tag_is_present(image, tags)
    repo_digests = item.get("RepoDigests")
    valid_digests = [
        value for value in repo_digests
        if isinstance(value, str) and _REPO_DIGEST.fullmatch(value)
    ] if isinstance(repo_digests, list) else []
    summary["repository_digest_valid"] = bool(valid_digests)
    summary["repository_digests"] = valid_digests
    summary["os"] = item.get("Os")
    summary["architecture"] = item.get("Architecture")

    failures = []
    if not summary["id_sha256_valid"]:
        failures.append("image ID is not a valid sha256 digest")
    if not summary["tag_matches"]:
        failures.append("requested tag is not present on the inspected image")
    if not summary["repository_digest_valid"]:
        failures.append("no valid repository sha256 digest is recorded")
    if summary["os"] != "linux":
        failures.append("image is not a Linux container image")
    summary["ready"] = not failures
    summary["error"] = "; ".join(failures) if failures else None
    return summary


def inspect_local_docker_images(
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Inspect the local daemon and required tags only; never pull or build."""
    executable = shutil.which("docker")
    if not executable:
        return {
            "engine_ready": False,
            "images_ready": False,
            "docker_version": None,
            "images": [
                {"image": name, "present": False, "ready": False, "error": "docker executable not found"}
                for name in REQUIRED_DOCKER_IMAGES
            ],
            "error": "Docker CLI is not installed or not on PATH.",
        }

    try:
        context = runner(
            [executable, "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=15, check=False, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        context = None
        context_error = str(exc)
    else:
        context_error = (context.stderr or context.stdout or "Docker context inspection failed.").strip()
    endpoint = (context.stdout or "").strip() if context is not None else ""
    if context is None or context.returncode != 0 or not endpoint.startswith(("unix://", "npipe://")):
        return {
            "engine_ready": False,
            "images_ready": False,
            "docker_version": None,
            "docker_endpoint": endpoint or None,
            "images": [
                {"image": name, "present": False, "ready": False, "error": "Local Docker endpoint not verified"}
                for name in REQUIRED_DOCKER_IMAGES
            ],
            "error": (
                "Docker must use a local unix:// socket or npipe:// named pipe; "
                "remote TCP/SSH endpoints are not accepted. "
                + (context_error[-1000:] if context_error else "Observed endpoint: " + endpoint)
            ),
        }

    try:
        info = runner(
            [executable, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=15, check=False, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "engine_ready": False,
            "images_ready": False,
            "docker_version": None,
            "docker_endpoint": endpoint,
            "images": [
                {"image": name, "present": False, "ready": False, "error": "Docker daemon unavailable"}
                for name in REQUIRED_DOCKER_IMAGES
            ],
            "error": "Cannot query the local Docker daemon: " + str(exc),
        }
    docker_version = (info.stdout or "").strip()
    if info.returncode != 0 or not docker_version:
        detail = (info.stderr or info.stdout or "Docker daemon did not return a server version.").strip()
        return {
            "engine_ready": False,
            "images_ready": False,
            "docker_version": None,
            "images": [
                {"image": name, "present": False, "ready": False, "error": "Docker daemon unavailable"}
                for name in REQUIRED_DOCKER_IMAGES
            ],
            "error": "Local Docker daemon is not ready: " + detail[-1000:],
        }

    images: list[dict[str, Any]] = []
    for image in REQUIRED_DOCKER_IMAGES:
        try:
            inspected = runner(
                [executable, "image", "inspect", image],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=20, check=False, shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            images.append({
                "image": image, "present": False, "ready": False,
                "error": "local image inspection failed: " + str(exc),
            })
            continue
        if inspected.returncode != 0:
            detail = (inspected.stderr or inspected.stdout or "Image is absent from the local store.").strip()
            images.append({
                "image": image, "present": False, "ready": False,
                "error": detail[-1000:],
            })
            continue
        try:
            payload = json.loads(inspected.stdout)
        except (json.JSONDecodeError, TypeError):
            images.append({
                "image": image, "present": False, "ready": False,
                "error": "docker image inspect returned invalid JSON",
            })
            continue
        images.append(validate_docker_image_inspect(image, payload))

    return {
        "engine_ready": True,
        "images_ready": all(item.get("ready") is True for item in images),
        "docker_version": docker_version,
        "images": images,
        "operation_policy": "local docker context inspect, docker info, and docker image inspect only; no pull, build, run, or remote endpoint is accepted",
        "error": None,
    }


def build_preflight_report(
    config: str | Path = "configs/training/phase4_milestone4.json",
    device: str | None = None,
    docker_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    profile = _load_profile(config)
    requested_device = device or profile.get("device", "auto")
    if requested_device not in {"auto", "cpu", "cuda", "mps"}:
        raise PreflightError("Requested training device must be auto, cpu, cuda, or mps.")
    torch_report = probe_torch_devices(requested_device)
    docker_report = inspect_local_docker_images(docker_runner)
    python_ready = bool(torch_report.get("python", {}).get("ready"))
    pytorch_ready = bool(torch_report.get("pytorch", {}).get("ready"))
    cpu_ready = bool(torch_report.get("cpu_required_ready"))
    device_ready = bool(torch_report.get("selected_device_ready"))
    docker_ready = bool(docker_report.get("engine_ready") and docker_report.get("images_ready"))
    return {
        "status": "GO" if all((python_ready, pytorch_ready, cpu_ready, device_ready, docker_ready)) else "NO-GO",
        "profile": profile.get("profile"),
        "requested_device": requested_device,
        "python_pytorch": torch_report,
        "docker": docker_report,
        "checks": {
            "python_version": python_ready,
            "pytorch_import": pytorch_ready,
            "cpu_tensor_execution": cpu_ready,
            "selected_training_device": device_ready,
            "local_docker_images_with_sha256": docker_ready,
        },
        "network_access_used": False,
        "docker_pull_or_build_performed": False,
    }


def _print_report(report: dict[str, Any]) -> None:
    print("SYNTRA LOCAL BASELINE PREFLIGHT")
    print("=" * 36)
    print(f"Overall: {report['status']}")
    print(f"Profile: {report.get('profile')}")
    print(f"Training device requested: {report.get('requested_device')}")
    python = report["python_pytorch"]["python"]
    pytorch = report["python_pytorch"]["pytorch"]
    print(f"Python {python['version']}: {'READY' if python['ready'] else 'UNAVAILABLE (Python 3.12+ required)'}")
    print(f"PyTorch {pytorch.get('version') or 'unavailable'}: {'READY' if pytorch['ready'] else 'UNAVAILABLE'}")
    for name, device in report["python_pytorch"]["devices"].items():
        state = "READY" if device.get("ready") else "AVAILABLE BUT FAILED" if device.get("available") else "NOT AVAILABLE"
        suffix = " — " + str(device.get("name")) if device.get("name") else ""
        print(f"  {name.upper():5} {state}{suffix}")
        if device.get("available") and not device.get("ready"):
            print("        " + str(device.get("detail")))
    docker = report["docker"]
    print(f"Docker engine: {'READY (' + str(docker.get('docker_version')) + ')' if docker.get('engine_ready') else 'UNAVAILABLE'}")
    for item in docker.get("images", []):
        state = "READY" if item.get("ready") else "MISSING / INVALID"
        digest_count = len(item.get("repository_digests", []))
        print(f"  {item['image']}: {state}; repository sha256 digests={digest_count}")
        if item.get("error"):
            print("        " + str(item["error"]))
    print("=" * 36)
    print("Network access: not used; Docker pull/build: not performed.")
    if report["status"] == "NO-GO":
        print("Do not start a long baseline run until every required check is ready.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify local PyTorch devices and pre-provisioned offline Docker images before SYNTRA training."
    )
    parser.add_argument(
        "--config", default="configs/training/phase4_milestone4.json",
        help="Training profile whose selected device should be checked.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), help="Override the profile's training device.")
    parser.add_argument("--json", action="store_true", help="Print the full machine-readable readiness report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_preflight_report(config=args.config, device=args.device)
    except (PreflightError, evaluate_epochs.OrchestrationError) as exc:
        report = {
            "status": "NO-GO",
            "error": str(exc),
            "network_access_used": False,
            "docker_pull_or_build_performed": False,
        }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        if "checks" not in report:
            print("SYNTRA LOCAL BASELINE PREFLIGHT")
            print("=" * 36)
            print("Overall: NO-GO")
            print(str(report.get("error", "Preflight did not complete.")))
        else:
            _print_report(report)
    return 0 if report.get("status") == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
