from __future__ import annotations

import hashlib
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import traceback


ROOT = pathlib.Path(__file__).resolve().parent
ENV_DIR = pathlib.Path(os.getenv("RF_ENV_DIR", "/content/renewed-fooocus-env"))
PYTHON = ENV_DIR / "bin" / "python"
REQUIREMENTS = ROOT / "requirements_versions.txt"
INSTALL_MARKER = ENV_DIR / ".renewed_fooocus_ready"
PYTHON_VERSION = "3.10"
TORCH_VERSION = "2.3.1"
TORCHVISION_VERSION = "0.18.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run(
    command: list[str],
    *,
    cwd: pathlib.Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [str(part) for part in command]
    print(f"\n$ {shlex.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def require_gpu() -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError(
            "No NVIDIA GPU was detected. In Colab choose Runtime > Change runtime type > T4 GPU, then run the cell again."
        )
    run([nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])


def uv_command() -> list[str]:
    executable = shutil.which("uv")
    if executable:
        return [executable]
    return [sys.executable, "-m", "uv"]


def environment_marker() -> str:
    digest = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    return (
        f"requirements={digest}\n"
        f"python={PYTHON_VERSION}\n"
        f"torch={TORCH_VERSION}\n"
        f"torchvision={TORCHVISION_VERSION}\n"
    )


def validate_environment(*, require_cuda: bool = True) -> bool:
    if not PYTHON.exists():
        return False

    smoke_test = """
import sys
import cv2
import gradio
import httpx
import numpy
import torch
import torchvision

assert sys.version_info[:2] == (3, 10), sys.version
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError('PyTorch cannot access the Colab GPU.')
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('Gradio:', gradio.__version__)
""".replace("REQUIRE_CUDA", "True" if require_cuda else "False")

    try:
        result = run([str(PYTHON), "-c", smoke_test], capture=True)
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, flush=True)
        if exc.stderr:
            print(exc.stderr, flush=True)
        return False

    if result.stdout:
        print(result.stdout.strip(), flush=True)
    return True


def create_environment(uv: list[str]) -> None:
    if ENV_DIR.exists():
        shutil.rmtree(ENV_DIR)

    # uv downloads a managed Python build when Colab does not provide Python 3.10.
    run(uv + ["python", "install", PYTHON_VERSION])
    run(uv + ["venv", "--python", PYTHON_VERSION, "--seed", str(ENV_DIR)])

    if not PYTHON.exists():
        raise RuntimeError(f"The Renewed Fooocus environment was not created at {ENV_DIR}.")


def install_environment() -> None:
    expected_marker = environment_marker()
    if (
        INSTALL_MARKER.exists()
        and INSTALL_MARKER.read_text(encoding="utf-8") == expected_marker
        and validate_environment(require_cuda=True)
    ):
        print("Renewed Fooocus environment is already installed and passed the CUDA smoke test.", flush=True)
        return

    print("Installing Renewed Fooocus in an isolated Python 3.10 environment...", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "uv"])
    uv = uv_command()
    create_environment(uv)

    run(
        uv
        + [
            "pip",
            "install",
            "--python",
            str(PYTHON),
            "--index-url",
            TORCH_INDEX,
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
        ]
    )
    run(uv + ["pip", "install", "--python", str(PYTHON), "-r", str(REQUIREMENTS)])

    if not validate_environment(require_cuda=True):
        raise RuntimeError(
            "The isolated environment installed, but its Python/CUDA import smoke test failed. "
            "Restart the Colab runtime and run the notebook again."
        )

    INSTALL_MARKER.write_text(expected_marker, encoding="utf-8")
    print("Environment installation and CUDA validation complete.", flush=True)


def verify_writable_directory(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".renewed_fooocus_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
    finally:
        probe.unlink(missing_ok=True)


def configure_storage(child_env: dict[str, str]) -> None:
    drive_root = os.getenv("RF_DRIVE_ROOT", "").strip()
    if not drive_root:
        print("Using temporary Colab storage.", flush=True)
        return

    persistent = pathlib.Path(drive_root).expanduser().resolve()
    paths = {
        "path_checkpoints": persistent / "models" / "checkpoints",
        "path_loras": persistent / "models" / "loras",
        "path_vae": persistent / "models" / "vae",
        "path_embeddings": persistent / "models" / "embeddings",
        "path_outputs": persistent / "outputs",
    }
    for key, path in paths.items():
        verify_writable_directory(path)
        child_env[key] = str(path)

    print(f"Persistent storage is writable: {persistent}", flush=True)


def split_references(value: str) -> list[str]:
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def download_civitai_assets(child_env: dict[str, str]) -> None:
    token = os.getenv("RF_CIVITAI_TOKEN", "").strip()
    if token:
        child_env["CIVITAI_API_TOKEN"] = token

    requested = {
        "Checkpoint": split_references(os.getenv("RF_CIVITAI_CHECKPOINTS", "")),
        "LORA": split_references(os.getenv("RF_CIVITAI_LORAS", "")),
    }
    destinations = {
        "Checkpoint": child_env.get("path_checkpoints", str(ROOT / "models" / "checkpoints")),
        "LORA": child_env.get("path_loras", str(ROOT / "models" / "loras")),
    }

    installer = r"""
import os
import sys
from modules.civitai_client import CivitaiError, install_reference

model_type, destination, *references = sys.argv[1:]
token = os.getenv('CIVITAI_API_TOKEN', '').strip() or None
failures = []
for reference in references:
    print(f'[Civitai] Resolving {model_type}: {reference}', flush=True)
    try:
        result = install_reference(
            reference=reference,
            destination_dir=destination,
            expected_type=model_type,
            api_token=token,
            overwrite=os.getenv('RF_CIVITAI_OVERWRITE', '').lower() in {'1', 'true', 'yes', 'on'},
        )
    except CivitaiError as exc:
        failures.append(f'{reference}: {exc}')
        print(f'[Civitai] Failed: {exc}', flush=True)
        continue
    record = result['record']
    print(f"[Civitai] Installed {record['model_name']} — {record['version_name']}", flush=True)
if failures:
    raise SystemExit('One or more requested Civitai downloads failed:\n' + '\n'.join(failures))
"""

    for model_type, references in requested.items():
        if not references:
            continue
        destination = pathlib.Path(destinations[model_type]).expanduser().resolve()
        verify_writable_directory(destination)
        run(
            [str(PYTHON), "-u", "-c", installer, model_type, str(destination), *references],
            cwd=ROOT,
            env=child_env,
        )


def launch_fooocus(child_env: dict[str, str]) -> None:
    preset = os.getenv("RF_PRESET", "realistic").strip()
    command = [str(PYTHON), "launch.py", "--share", "--always-high-vram"]
    if preset:
        command.extend(["--preset", preset])

    child_env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    print("\nStarting Renewed Fooocus.", flush=True)
    print("Open the gradio.live URL printed below when startup completes.\n", flush=True)
    run(command, cwd=ROOT, env=child_env)


def main() -> int:
    try:
        require_gpu()
        free_gb = shutil.disk_usage("/content").free / 1024**3
        print(f"Free Colab storage: {free_gb:.1f} GB", flush=True)
        if free_gb < 12:
            raise RuntimeError("Less than 12 GB of free Colab storage is available. Restart the runtime or remove large files.")

        install_environment()
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        child_env.setdefault("UV_LINK_MODE", "copy")
        configure_storage(child_env)
        download_civitai_assets(child_env)
        launch_fooocus(child_env)
        return 0
    except KeyboardInterrupt:
        print("Renewed Fooocus was stopped.", flush=True)
        return 130
    except Exception as exc:
        print("\n" + "=" * 72, flush=True)
        print(f"RENEWED FOOOCUS STARTUP FAILED: {exc}", flush=True)
        print("=" * 72, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
