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
ENV_DIR = pathlib.Path("/content/renewed-fooocus-env")
PYTHON = ENV_DIR / "bin" / "python"
REQUIREMENTS = ROOT / "requirements_versions.txt"
INSTALL_MARKER = ENV_DIR / ".renewed_fooocus_ready"


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run(command: list[str], *, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None) -> None:
    command = [str(part) for part in command]
    print(f"\n$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_gpu() -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError(
            "No NVIDIA GPU was detected. In Colab choose Runtime > Change runtime type > T4 GPU, then run the cell again."
        )
    run([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"])


def uv_command() -> list[str]:
    executable = shutil.which("uv")
    if executable:
        return [executable]
    return [sys.executable, "-m", "uv"]


def create_environment(uv: list[str]) -> None:
    if ENV_DIR.exists():
        shutil.rmtree(ENV_DIR)

    try:
        run(uv + ["venv", "--python", "3.10", "--seed", str(ENV_DIR)])
    except subprocess.CalledProcessError:
        print(
            "Python 3.10 provisioning failed. Falling back to Colab's current Python runtime.",
            flush=True,
        )
        run([sys.executable, "-m", "venv", "--clear", str(ENV_DIR)])

    if not PYTHON.exists():
        raise RuntimeError(f"The Renewed Fooocus environment was not created at {ENV_DIR}.")


def install_environment() -> None:
    digest = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    expected_marker = f"{digest}\ntorch=2.3.1\ntorchvision=0.18.1\n"

    if PYTHON.exists() and INSTALL_MARKER.exists() and INSTALL_MARKER.read_text() == expected_marker:
        print("Renewed Fooocus environment is already installed for this runtime.", flush=True)
        return

    print("Installing Renewed Fooocus in an isolated environment...", flush=True)
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
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu121",
            "torch==2.3.1",
            "torchvision==0.18.1",
        ]
    )
    run(uv + ["pip", "install", "--python", str(PYTHON), "-r", str(REQUIREMENTS)])
    INSTALL_MARKER.write_text(expected_marker)
    print("Environment installation complete.", flush=True)


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
        path.mkdir(parents=True, exist_ok=True)
        child_env[key] = str(path)

    print(f"Persistent storage: {persistent}", flush=True)


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

    for model_type, references in requested.items():
        if not references:
            continue
        command = [
            str(PYTHON),
            "-u",
            "-m",
            "modules.civitai_client",
            "--type",
            model_type,
            "--destination",
            destinations[model_type],
        ]
        if env_flag("RF_CIVITAI_OVERWRITE", False):
            command.append("--overwrite")
        for reference in references:
            command.extend(["--reference", reference])
        run(command, cwd=ROOT, env=child_env)


def launch_fooocus(child_env: dict[str, str]) -> None:
    preset = os.getenv("RF_PRESET", "realistic").strip()
    command = [str(PYTHON), "launch.py", "--share", "--always-high-vram"]
    if preset:
        command.extend(["--preset", preset])

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
