from __future__ import annotations

import hashlib
import os
import pathlib
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time


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
            "No NVIDIA GPU was detected. In Colab choose Runtime > Change runtime type > T4 GPU, then run the cell again."
        )
    run([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"])


def install_environment() -> None:
    digest = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    expected_marker = f"{digest}\ntorch=2.3.1\ntorchvision=0.18.1\n"

    if PYTHON.exists() and INSTALL_MARKER.exists() and INSTALL_MARKER.read_text() == expected_marker:
        print("Renewed Fooocus environment is already installed for this runtime.", flush=True)
        return

    print("Installing the isolated Python 3.10 environment...", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "uv"])
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv installed but its executable could not be located.")

    if ENV_DIR.exists():
        shutil.rmtree(ENV_DIR)

    run([uv, "venv", "--python", "3.10", "--seed", str(ENV_DIR)])
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(PYTHON),
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
            "torch==2.3.1",
            "torchvision==0.18.1",
        ]
    )
    run([uv, "pip", "install", "--python", str(PYTHON), "-r", str(REQUIREMENTS)])
    INSTALL_MARKER.write_text(expected_marker)
    print("Environment installation complete.", flush=True)


def configure_storage(child_env: dict[str, str]) -> None:
    if not env_flag("RF_SAVE_TO_DRIVE", False):
        print("Using temporary Colab storage. Enable SAVE_TO_GOOGLE_DRIVE to keep models and outputs.", flush=True)
        return

    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Google Drive persistence is available only inside Google Colab.") from exc

    print("Mounting Google Drive...", flush=True)
    drive.mount("/content/drive", force_remount=False)
    persistent = pathlib.Path("/content/drive/MyDrive/Renewed Fooocus")

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


def wait_for_share_url(process: subprocess.Popen, log_path: pathlib.Path, timeout_seconds: int = 75) -> str | None:
    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.gradio\.live")
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(1)
        text = log_path.read_text(errors="ignore") if log_path.exists() else ""
        match = pattern.search(text)
        if match:
            return match.group(0)
        if process.poll() is not None:
            print("Civitai Manager stopped during startup. Its log follows:\n", flush=True)
            print(text[-5000:], flush=True)
            return None
    return None


def start_civitai_manager(child_env: dict[str, str]) -> subprocess.Popen | None:
    if not env_flag("RF_START_CIVITAI", True):
        print("Civitai Manager disabled for this run.", flush=True)
        return None

    username = os.getenv("RF_CIVITAI_USERNAME", "renewed").strip() or "renewed"
    password = os.getenv("RF_CIVITAI_PASSWORD", "").strip() or secrets.token_urlsafe(12)
    manager_env = child_env.copy()
    manager_env.update(
        {
            "FOOOCUS_CIVITAI_HOST": "0.0.0.0",
            "FOOOCUS_CIVITAI_PORT": "7866",
            "FOOOCUS_CIVITAI_SHARE": "1",
            "FOOOCUS_CIVITAI_USERNAME": username,
            "FOOOCUS_CIVITAI_PASSWORD": password,
            "PYTHONUNBUFFERED": "1",
        }
    )

    log_path = ROOT / "civitai_manager.log"
    log_handle = log_path.open("w")
    process = subprocess.Popen(
        [str(PYTHON), "civitai_manager.py"],
        cwd=ROOT,
        env=manager_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    process._renewed_log_handle = log_handle  # type: ignore[attr-defined]

    public_url = wait_for_share_url(process, log_path)
    print("\n" + "=" * 72, flush=True)
    if public_url:
        print(f"CIVITAI MANAGER: {public_url}", flush=True)
        print(f"USERNAME: {username}", flush=True)
        print(f"PASSWORD: {password}", flush=True)
    else:
        print(f"Civitai Manager did not produce a public link. Fooocus will still start. Log: {log_path}", flush=True)
    print("=" * 72 + "\n", flush=True)
    return process


def launch_fooocus(child_env: dict[str, str]) -> None:
    preset = os.getenv("RF_PRESET", "realistic").strip()
    command = [str(PYTHON), "launch.py", "--share", "--always-high-vram"]
    if preset:
        command.extend(["--preset", preset])

    print("Starting Renewed Fooocus. The first run downloads one SDXL checkpoint and may take several minutes.", flush=True)
    print("Open the gradio.live URL shown below when startup completes.\n", flush=True)
    run(command, cwd=ROOT, env=child_env)


def main() -> None:
    require_gpu()
    free_gb = shutil.disk_usage("/content").free / 1024**3
    print(f"Free Colab storage: {free_gb:.1f} GB", flush=True)
    if free_gb < 15:
        raise RuntimeError("Less than 15 GB of free Colab storage is available. Restart the runtime or remove large files.")

    install_environment()
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    configure_storage(child_env)
    manager = start_civitai_manager(child_env)

    try:
        launch_fooocus(child_env)
    finally:
        if manager and manager.poll() is None:
            manager.terminate()
        if manager:
            log_handle = getattr(manager, "_renewed_log_handle", None)
            if log_handle:
                log_handle.close()


if __name__ == "__main__":
    main()
