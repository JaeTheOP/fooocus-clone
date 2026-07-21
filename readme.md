<div align="center">

# Renewed Fooocus

### One-cell Google Colab image generation with direct Civitai model downloads

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaeTheOP/fooocus-clone/blob/agent/civitai-model-manager/colab.ipynb)
[![Based on Fooocus](https://img.shields.io/badge/based%20on-Fooocus-5b6cff)](https://github.com/lllyasviel/Fooocus)
[![License](https://img.shields.io/github/license/JaeTheOP/fooocus-clone)](LICENSE)

**Open the notebook. Select a GPU. Run one cell.**

</div>

---

## What is Renewed Fooocus?

**Renewed Fooocus** is an independent community continuation of the original [Fooocus](https://github.com/lllyasviel/Fooocus) SDXL image-generation interface.

It preserves the familiar Fooocus workflow while adding:

- A one-cell Google Colab launcher
- A managed Python 3.10 environment that does not replace Colab's system Python
- A startup smoke test for Python, PyTorch, CUDA, Gradio, OpenCV, HTTPX, and NumPy
- Optional Google Drive persistence for models and outputs
- Direct Civitai checkpoint and LoRA installation from exact version URLs or IDs
- Automatic installation into the correct model folders
- A separate local Civitai search-and-install manager for desktop or server use
- Renewed Fooocus branding in the browser and companion tools

> **Independent fork:** Renewed Fooocus is not the official Fooocus project and is not affiliated with Civitai. Upstream attribution is provided below.

## Verified one-cell Google Colab

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaeTheOP/fooocus-clone/blob/agent/civitai-model-manager/colab.ipynb)

### Run it

1. Open the Colab notebook.
2. Choose **Runtime → Change runtime type**.
3. Select **T4 GPU** or another available NVIDIA GPU.
4. Optionally enable **SAVE_TO_GOOGLE_DRIVE**.
5. Optionally enter exact Civitai checkpoint or LoRA version URLs or numeric version IDs.
6. Run the single **▶ Run Renewed Fooocus** cell.

The cell performs the complete startup process:

- Retries the repository clone if GitHub has a transient connection failure
- Prints the exact Renewed Fooocus commit being run
- Verifies that an NVIDIA GPU is available
- Checks available Colab storage
- Installs `uv`
- Downloads a managed Python 3.10 build
- Creates an isolated virtual environment
- Installs the official PyTorch 2.3.1 CUDA 12.1 wheels and pinned dependencies
- Verifies that PyTorch can access the allocated GPU
- Optionally mounts and verifies writable Google Drive folders
- Installs requested Civitai assets before launch
- Stops with a clear error when a required startup check or requested download fails
- Launches one Renewed Fooocus `gradio.live` interface

Keep the cell running while using Renewed Fooocus. Stopping the cell shuts down the temporary public interface.

### Successful startup output

Before the interface launches, the notebook prints information similar to:

```text
NVIDIA T4, 15360 MiB, <driver version>
Python: 3.10.x
PyTorch: 2.3.1+cu121
Torchvision: 0.18.1+cu121
CUDA available: True
GPU: Tesla T4
Gradio: 3.41.2
```

Renewed Fooocus then prints a temporary URL similar to:

```text
https://example.gradio.live
```

Open that URL in a new tab. Gradio share URLs expire when the Colab process or runtime stops.

### First startup

The default Colab preset is `realistic`. On the first launch, the notebook installs the Python environment and Fooocus downloads the assets required by the selected preset. Later reruns in the same active runtime reuse the validated Python environment.

Leave the Civitai fields empty for the fastest first launch. Add custom checkpoints or LoRAs after confirming that the default interface starts correctly.

## Direct Civitai downloads in Colab

The one-cell notebook accepts:

- An exact Civitai model URL containing `modelVersionId=...`
- A Civitai API download URL
- A numeric Civitai model-version ID
- Multiple references separated by commas or line breaks

Example checkpoint field:

```text
https://civitai.com/models/123/model-name?modelVersionId=456, 789
```

Put checkpoint versions in **CIVITAI_CHECKPOINTS** and LoRA versions in **CIVITAI_LORAS**. The launcher verifies that the resolved asset type matches the field where it was entered.

An API token is only needed for account-restricted or gated Civitai downloads. The notebook passes the token to the child process and does not intentionally write it to the repository or Google Drive.

Requested Civitai downloads are treated as required. When one fails, startup stops and reports which reference failed instead of silently opening Fooocus without the requested model.

## Google Drive persistence

Enable `SAVE_TO_GOOGLE_DRIVE` to store files under:

```text
MyDrive/Renewed Fooocus/
├── models/
│   ├── checkpoints/
│   ├── loras/
│   ├── vae/
│   └── embeddings/
└── outputs/
```

The launcher creates each directory, verifies that it is writable, and exports it through Fooocus's supported path environment variables.

When Drive persistence is disabled, models and outputs stored under `/content` disappear when the Colab runtime is deleted or reset.

## Model download protections

The Civitai downloader:

- Accepts `.safetensors` checkpoint and LoRA files
- Rejects failed Civitai virus or pickle scan results
- Validates download hosts, filenames, and destination paths
- Uses temporary `.part` files and atomic replacement
- Enforces a configurable maximum download size
- Verifies SHA-256 when Civitai supplies a full hash
- Writes a `.civitai.json` metadata sidecar
- Keeps the optional API token in memory

The default maximum download size is 12 GB. For local use it can be changed with:

```bash
FOOOCUS_CIVITAI_MAX_GB=20 python civitai_manager.py
```

## Local installation

### Requirements

- Python 3.10
- Git
- An NVIDIA GPU is recommended
- At least 4 GB VRAM for basic use
- Enough storage for SDXL checkpoints and outputs

### Linux or macOS

```bash
git clone --branch agent/civitai-model-manager https://github.com/JaeTheOP/fooocus-clone.git renewed-fooocus
cd renewed-fooocus
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_versions.txt
python launch.py
```

### Windows

```powershell
git clone --branch agent/civitai-model-manager https://github.com/JaeTheOP/fooocus-clone.git renewed-fooocus
cd renewed-fooocus
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_versions.txt
python launch.py
```

The correct PyTorch command may differ for local GPUs and drivers. Use the official PyTorch installation selector when CUDA 12.1 is not appropriate for the machine.

## Run the local Civitai Manager

The separate search interface remains available for local installations:

```bash
python civitai_manager.py
```

Default address:

```text
http://127.0.0.1:7866
```

Create an authenticated temporary public link:

```bash
FOOOCUS_CIVITAI_SHARE=1 \
FOOOCUS_CIVITAI_USERNAME=renewed \
FOOOCUS_CIVITAI_PASSWORD='replace-this-password' \
python civitai_manager.py
```

Do not expose the manager publicly without authentication. Anyone with access could initiate large downloads into the environment.

## Automated validation

The repository includes a GitHub Actions workflow that:

- Compiles the Colab launcher and Civitai modules
- Parses the notebook as valid notebook JSON
- Compiles every Colab code cell with Python's AST parser
- Runs the Civitai client unit tests
- Runs the Colab launcher storage, input-parsing, and launch-command tests

These checks catch syntax, notebook-format, path-export, command-construction, and Civitai parsing regressions. A true end-to-end GPU launch still requires a Google Colab GPU runtime because GitHub's standard CI runners do not provide the same NVIDIA environment.

## Core capabilities retained

Renewed Fooocus retains the primary workflows from upstream Fooocus:

- Text-to-image generation
- Prompt expansion and style presets
- SDXL checkpoint and refiner selection
- Multiple LoRAs with adjustable weights
- Image variation and upscaling
- Inpainting and outpainting
- Image Prompt, FaceSwap, PyraCanny, and CPDS controls
- Image description tools
- Metadata import and export
- Advanced sampler, scheduler, and guidance controls

## Compatibility

Renewed Fooocus is primarily designed for the **SDXL model family**. Downloading a model does not make an incompatible architecture usable by Fooocus. Flux, SD 1.5, Stable Cascade, and other architectures may require different software.

Review each model's license and usage terms before use or redistribution.

## Troubleshooting

### No GPU was detected

Choose **Runtime → Change runtime type → T4 GPU**, save the setting, and rerun the cell. The launcher intentionally refuses to proceed on a CPU-only runtime.

### The CUDA smoke test failed

Choose **Runtime → Disconnect and delete runtime**, reconnect with a GPU, and rerun the notebook. The launcher rebuilds the managed Python environment when its validation marker or imports are invalid.

### Less than 12 GB of free Colab storage is available

Select **Runtime → Disconnect and delete runtime**, reconnect, and rerun the notebook. Remove large files from `/content` when reusing a session.

### The first startup takes a while

The first run installs the CUDA Python environment and downloads the selected preset assets. Later reruns in the same active runtime reuse the validated environment.

### A model does not appear

Open **Advanced → Models → Refresh All Files**. Confirm that the asset is an SDXL-compatible `.safetensors` checkpoint or LoRA.

### A Civitai download failed

Confirm that the value is an exact model-version URL or numeric version ID, that the asset type matches the field, and that an API token is supplied when the download requires an account.

### A public Gradio URL is not created

Gradio share links depend on outbound network access and are temporary. Read the final error block printed in the cell output. The launcher now stops on failed environment validation rather than opening a partially working session.

## Upstream and licensing

Renewed Fooocus is derived from [Fooocus by lllyasviel and its contributors](https://github.com/lllyasviel/Fooocus).

The upstream project created the core generation pipeline, prompt processing, inpainting, image-prompt systems, presets, and user interface on which this fork is based.

This repository retains the upstream license. See [`LICENSE`](LICENSE) and the repository history for copyright and contributor details.

- Original project: [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus)
- Renewed fork: [JaeTheOP/fooocus-clone](https://github.com/JaeTheOP/fooocus-clone)
- Civitai integration: [`CIVITAI_MANAGER.md`](CIVITAI_MANAGER.md)

## Disclaimer

You are responsible for the models you download, their licenses, the prompts and source images you provide, and the outputs you create. Do not use the software to violate applicable law, privacy rights, intellectual-property rights, platform rules, or model licenses.
