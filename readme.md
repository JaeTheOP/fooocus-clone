<div align="center">

# Renewed Fooocus

### One-click Google Colab image generation with direct Civitai model management

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaeTheOP/fooocus-clone/blob/agent/civitai-model-manager/colab.ipynb)
[![Based on Fooocus](https://img.shields.io/badge/based%20on-Fooocus-5b6cff)](https://github.com/lllyasviel/Fooocus)
[![License](https://img.shields.io/github/license/JaeTheOP/fooocus-clone)](LICENSE)

**Open the notebook. Select a GPU. Run one cell.**

</div>

---

## What is Renewed Fooocus?

**Renewed Fooocus** is an independent community continuation of the original [Fooocus](https://github.com/lllyasviel/Fooocus) image-generation interface.

It retains the familiar Fooocus SDXL workflow while adding:

- A single-cell Google Colab launcher
- An isolated Python 3.10 environment that does not replace Colab's system Python
- Optional Google Drive persistence
- Direct Civitai checkpoint and LoRA search
- Specific model-version selection
- Automatic installation into the correct model folders
- A password-protected public Civitai Manager
- Renewed Fooocus branding in the browser and companion tools

> **Independent fork:** Renewed Fooocus is not the official Fooocus project and is not affiliated with Civitai. Upstream attribution is provided below.

## One-click Google Colab

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaeTheOP/fooocus-clone/blob/agent/civitai-model-manager/colab.ipynb)

### Run it

1. Open the Colab notebook.
2. Choose **Runtime → Change runtime type**.
3. Select **T4 GPU** or another available NVIDIA GPU.
4. Optionally enable **SAVE_TO_GOOGLE_DRIVE**.
5. Leave **START_CIVITAI_MANAGER** enabled to use direct Civitai downloads.
6. Run the single **▶ Run Renewed Fooocus** cell.

The cell handles the complete process:

- Verifies that a GPU is available
- Checks available Colab storage
- Clones the current Renewed Fooocus branch
- Installs `uv`
- Creates a managed Python 3.10 virtual environment
- Installs the CUDA PyTorch build and pinned dependencies
- Optionally mounts Google Drive
- Starts the authenticated Civitai Manager
- Launches Renewed Fooocus

Keep the cell running while using Renewed Fooocus. Stopping the cell shuts down the public interfaces.

### Links printed by Colab

The launcher prints two separate links:

```text
CIVITAI MANAGER: https://....gradio.live
USERNAME: renewed
PASSWORD: generated-password
```

Renewed Fooocus then prints its own separate `gradio.live` URL. Open that second URL to generate images.

### First startup

The default Colab preset is `realistic`. It downloads one SDXL checkpoint and one lightweight LoRA on first launch.

The older notebook used the `photoreal_civitai` preset, which attempted to download several complete checkpoints at startup. That behavior has been removed from the default Colab workflow because it could exhaust storage or make startup appear frozen.

### Google Drive persistence

Enable `SAVE_TO_GOOGLE_DRIVE` in the single Colab cell to store files under:

```text
MyDrive/Renewed Fooocus/
├── models/
│   ├── checkpoints/
│   ├── loras/
│   ├── vae/
│   └── embeddings/
└── outputs/
```

When Drive persistence is disabled, models and outputs are deleted when the Colab runtime is reset.

## Using the Civitai Manager

The companion manager can:

- Search Civitai checkpoints and LoRAs
- Sort by downloads, rating, or newest release
- Select an exact model version
- Default to SDXL-family assets compatible with Fooocus
- Display architecture, file size, compatibility, and trigger words
- Install files into the configured checkpoint or LoRA folder
- Use an optional Civitai API token for account-restricted assets

After installing a model:

1. Open Renewed Fooocus.
2. Enable **Advanced**.
3. Open **Models**.
4. Click **Refresh All Files**.
5. Select the new checkpoint or LoRA.

## Model download protections

The Civitai downloader:

- Accepts `.safetensors` model files
- Rejects failed Civitai virus or pickle scan results
- Validates filenames and destination paths
- Uses temporary `.part` downloads and atomic replacement
- Enforces a configurable maximum download size
- Verifies SHA-256 when a full hash is supplied
- Writes a `.civitai.json` metadata sidecar
- Keeps the optional API token in memory

The default maximum download size is 12 GB. It can be changed with:

```bash
FOOOCUS_CIVITAI_MAX_GB=20 python civitai_manager.py
```

## Local installation

### Requirements

- Python 3.10
- Git
- An NVIDIA GPU is recommended
- At least 4 GB VRAM for basic use
- Enough storage for SDXL checkpoints

### Linux or macOS

```bash
git clone --branch agent/civitai-model-manager https://github.com/JaeTheOP/fooocus-clone.git renewed-fooocus
cd renewed-fooocus
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_versions.txt
python entry_with_update.py
```

### Windows

```powershell
git clone --branch agent/civitai-model-manager https://github.com/JaeTheOP/fooocus-clone.git renewed-fooocus
cd renewed-fooocus
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_versions.txt
python entry_with_update.py
```

PyTorch installation may need to be adjusted for the CUDA version supported by the local machine.

## Run the Civitai Manager locally

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

Renewed Fooocus is primarily designed for the **SDXL model family**. The Civitai Manager defaults to assets likely to work with this pipeline.

An all-architectures download mode may show other model families, but downloading a file does not make it compatible with Renewed Fooocus. Flux, SD 1.5, Stable Cascade, and other architectures may require different software.

Review each model's license and usage terms before use or redistribution.

## Troubleshooting

### Colab says no GPU was detected

Choose **Runtime → Change runtime type → T4 GPU**, save the setting, and rerun the cell.

### Colab has less than 15 GB free

Select **Runtime → Disconnect and delete runtime**, reconnect with a GPU, and run the notebook again. Remove large files from `/content` when reusing a session.

### The first startup takes a while

The first launch installs dependencies and downloads an SDXL checkpoint. Later reruns in the same active runtime reuse the installed Python environment, although the notebook reclones the small source repository.

### A model does not appear

Open **Advanced → Models → Refresh All Files**. Confirm that the asset is an SDXL-compatible `.safetensors` checkpoint or LoRA.

### Civitai requires an account

Create a Civitai API token in your account settings and paste it into the optional token field. The manager does not intentionally save it.

### A public Gradio URL is not created

Gradio share links depend on outbound network access and are temporary. Inspect the cell output for the exact error. Renewed Fooocus and the Civitai Manager use separate public links.

## Upstream and licensing

Renewed Fooocus is derived from [Fooocus by lllyasviel and its contributors](https://github.com/lllyasviel/Fooocus).

The upstream project created the core generation pipeline, prompt processing, inpainting, image-prompt systems, presets, and user interface on which this fork is based.

This repository retains the upstream license. See [`LICENSE`](LICENSE) and the repository history for copyright and contributor details.

- Original project: [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus)
- Renewed fork: [JaeTheOP/fooocus-clone](https://github.com/JaeTheOP/fooocus-clone)
- Civitai integration: [`CIVITAI_MANAGER.md`](CIVITAI_MANAGER.md)

## Disclaimer

You are responsible for the models you download, their licenses, the prompts and source images you provide, and the outputs you create. Do not use the software to violate applicable law, privacy rights, intellectual-property rights, platform rules, or model licenses.
