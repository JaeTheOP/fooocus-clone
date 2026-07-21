from __future__ import annotations

import hashlib
import os
import pathlib
import shlex
import shutil
import subprocess
import sys


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
    print(f"\n$ {shlex.join(str(part) for part in command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_gpu() -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError(
            "No NVIDIA GPU was detected. In Colab choose Runtime > Change runtime type > T4 GPU, then run again."
        )
    run([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"])


def install_environment() -> None:
    digest = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    expected_marker = f"{digest}\ntorch=2.3.1\ntorchvision=0.18.1\n"

    if PYTHON.exists() and INSTALL_MARKER.exists() and INSTALL_MARKER.read_text() == expected_marker:
        print("Renewed Fooocus is already installed for this runtime.", flush=True)
        return

    print("Installing Renewed Fooocus in an isolated Python 3.10 environment...", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "uv"])
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv installed but its executable could not be located.")

    if ENV_DIR.exists():
        shutil.rmtree(ENV_DIR)

    run([uv, "venv", "--python", "3.10", "--seed", str(ENV_DIR)])
    run([
        uv, "pip", "install", "--python", str(PYTHON),
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "torch==2.3.1", "torchvision==0.18.1",
    ])
    run([uv, "pip", "install", "--python", str(PYTHON), "-r", str(REQUIREMENTS)])
    INSTALL_MARKER.write_text(expected_marker)
    print("Installation complete.", flush=True)


def configure_storage(child_env: dict[str, str]) -> tuple[pathlib.Path, pathlib.Path]:
    if env_flag("RF_SAVE_TO_DRIVE", True):
        try:
            from google.colab import drive
        except ImportError as exc:
            raise RuntimeError("Google Drive persistence is available only inside Google Colab.") from exc

        print("Mounting Google Drive...", flush=True)
        drive.mount("/content/drive", force_remount=False)
        persistent = pathlib.Path("/content/drive/MyDrive/Renewed Fooocus")
        checkpoints = persistent / "models" / "checkpoints"
        loras = persistent / "models" / "loras"
        paths = {
            "path_checkpoints": checkpoints,
            "path_loras": loras,
            "path_vae": persistent / "models" / "vae",
            "path_embeddings": persistent / "models" / "embeddings",
            "path_outputs": persistent / "outputs",
        }
        for key, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            child_env[key] = str(path)
        print(f"Persistent storage: {persistent}", flush=True)
        return checkpoints, loras

    checkpoints = ROOT / "models" / "checkpoints"
    loras = ROOT / "models" / "loras"
    checkpoints.mkdir(parents=True, exist_ok=True)
    loras.mkdir(parents=True, exist_ok=True)
    print("Using temporary Colab storage.", flush=True)
    return checkpoints, loras


def split_references(value: str) -> list[str]:
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def install_civitai_assets(child_env: dict[str, str], checkpoints: pathlib.Path, loras: pathlib.Path) -> None:
    token = os.getenv("RF_CIVITAI_TOKEN", "").strip()
    download_env = child_env.copy()
    if token:
        download_env["CIVITAI_API_TOKEN"] = token

    groups = [
        ("Checkpoint", checkpoints, split_references(os.getenv("RF_CIVITAI_CHECKPOINTS", ""))),
        ("LORA", loras, split_references(os.getenv("RF_CIVITAI_LORAS", ""))),
    ]

    for model_type, destination, references in groups:
        if not references:
            continue
        command = [
            str(PYTHON), "-m", "modules.civitai_client",
            "--type", model_type,
            "--destination", str(destination),
        ]
        for reference in references:
            command.extend(["--reference", reference])
        run(command, cwd=ROOT, env=download_env)


def launch_fooocus(child_env: dict[str, str]) -> None:
    preset = os.getenv("RF_PRESET", "realistic").strip()
    command = [str(PYTHON), "launch.py", "--share", "--always-high-vram"]
    if preset and preset != "default":
        command.extend(["--preset", preset])

    print("\n" + "=" * 72, flush=True)
    print("Renewed Fooocus is starting.", flush=True)
    print("Open the gradio.live URL printed below when startup finishes.", flush=True)
    print("Keep this Colab cell running while using the app.", flush=True)
    print("=" * 72 + "\n", flush=True)
    run(command, cwd=ROOT, env=child_env)


def main() -> None:
    require_gpu()
    free_gb = shutil.disk_usage("/content").free / 1024**3
    print(f"Free Colab storage: {free_gb:.1f} GB", flush=True)
    if free_gb < 12:
        raise RuntimeError("Less than 12 GB of free Colab storage is available. Restart the runtime or remove files.")

    install_environment()
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    checkpoints, loras = configure_storage(child_env)
    install_civitai_assets(child_env, checkpoints, loras)
    launch_fooocus(child_env)


if __name__ == "__main__":
    main()
