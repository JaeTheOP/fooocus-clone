<div align="center">

# Renewed Fooocus

### A modern, Colab-ready continuation of Fooocus with direct Civitai model management

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaeTheOP/fooocus-clone/blob/agent/civitai-model-manager/colab.ipynb)
[![Upstream](https://img.shields.io/badge/based%20on-Fooocus-5b6cff)](https://github.com/lllyasviel/Fooocus)
[![License](https://img.shields.io/github/license/JaeTheOP/fooocus-clone)](LICENSE)

**Prompt. Create. Install new models without leaving the interface.**

</div>

---

## What is Renewed Fooocus?

**Renewed Fooocus** is an independent community continuation of the original [Fooocus](https://github.com/lllyasviel/Fooocus) image-generation interface.

It keeps the simple Fooocus workflow while adding a maintained Google Colab setup and a built-in companion interface for finding and installing compatible Civitai checkpoints and LoRAs.

This fork is designed for users who want:

- A straightforward SDXL image-generation interface
- A Google Colab notebook that works without modifying Colab's system Python
- Optional Google Drive persistence for models and generated images
- Direct Civitai checkpoint and LoRA discovery
- Version-level model selection instead of manually copying download URLs
- Automatic installation into the correct Fooocus model folders
- The familiar Fooocus prompt, image-prompt, inpaint, outpaint, upscale, style, and preset workflow

> **Independent fork:** Renewed Fooocus is not the official Fooocus project and is not affiliated with Civitai. The original Fooocus authors and contributors remain credited under [Upstream and licensing](#upstream-and-licensing).

## Highlights

### Renewed Google Colab workflow

The included Colab notebook:

- Creates an isolated Python 3.10 environment with Micromamba
- Installs a CUDA-compatible PyTorch build
- Leaves Colab's system Python environment intact
- Supports T4 and other compatible NVIDIA Colab GPUs
- Can mount Google Drive for persistent checkpoints, LoRAs, VAEs, embeddings, and outputs
- Starts Renewed Fooocus and the Civitai Manager as separate Gradio interfaces
- Protects the public Civitai Manager link with generated credentials

### Direct Civitai model manager

The companion manager can:

- Search Civitai for checkpoints or LoRAs
- Sort by downloads, rating, or newest release
- Select a specific model version
- Default to Fooocus-compatible SDXL-family assets
- Show architecture, download size, compatibility, and trigger words
- Install directly into the configured checkpoint or LoRA folder
- Refresh Renewed Fooocus's local model index after installation
- Use an optional Civitai API token for account-restricted downloads

### Safer model installation

The downloader:

- Accepts `.safetensors` model files
- Rejects files with failed Civitai virus or pickle scan results
- Validates destination paths and filenames
- Uses atomic `.part` downloads
- Enforces a configurable maximum download size
- Verifies SHA-256 when Civitai provides a full hash
- Stores model metadata in a sidecar `.civitai.json` file
- Keeps the optional API token in memory rather than writing it to disk

## Quick start: Google Colab

Open the notebook:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaeTheOP/fooocus-clone/blob/agent/civitai-model-manager/colab.ipynb)

Then run the notebook cells in order:

1. **Install Renewed Fooocus** in an isolated Python 3.10 environment.
2. **Choose Google Drive persistence.** Keep it enabled when you want models and outputs to survive runtime resets.
3. **Start the Civitai Manager.** Colab prints its public URL, username, and generated password.
4. **Launch Renewed Fooocus.** Open the separate `gradio.live` URL printed by the launcher.

After installing a model through the manager:

1. Open **Advanced → Models** in Renewed Fooocus.
2. Click **Refresh All Files**.
3. Select the new checkpoint or LoRA.

### Colab storage

With Drive persistence enabled, files are stored under:

```text
MyDrive/Renewed Fooocus/
├── models/
│   ├── checkpoints/
│   ├── loras/
│   ├── vae/
│   └── embeddings/
└── outputs/
```

Large checkpoints consume Drive storage quickly. Check the displayed download size before installing a model.

## Local installation

### Requirements

- Python 3.10
- Git
- An NVIDIA GPU is recommended
- At least 4 GB of VRAM for basic use; more VRAM is recommended for larger workflows
- Enough free disk space for checkpoints and generated images

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

For a remotely accessible interface:

```bash
python entry_with_update.py --listen
```

### Windows

Clone or download the repository, install Python 3.10, create a virtual environment, and install the pinned requirements:

```powershell
git clone --branch agent/civitai-model-manager https://github.com/JaeTheOP/fooocus-clone.git renewed-fooocus
cd renewed-fooocus
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_versions.txt
python entry_with_update.py
```

PyTorch installation may vary by GPU and CUDA version. Use the appropriate command from the official PyTorch installation selector when the default environment does not match your hardware.

## Run the Civitai Manager locally

From the repository root:

```bash
python civitai_manager.py
```

The default local address is:

```text
http://127.0.0.1:7866
```

To expose it on your network:

```bash
FOOOCUS_CIVITAI_HOST=0.0.0.0 python civitai_manager.py
```

To create a temporary public Gradio link with authentication:

```bash
FOOOCUS_CIVITAI_SHARE=1 \
FOOOCUS_CIVITAI_USERNAME=renewed \
FOOOCUS_CIVITAI_PASSWORD='replace-this-password' \
python civitai_manager.py
```

On Windows PowerShell:

```powershell
$env:FOOOCUS_CIVITAI_SHARE="1"
$env:FOOOCUS_CIVITAI_USERNAME="renewed"
$env:FOOOCUS_CIVITAI_PASSWORD="replace-this-password"
python civitai_manager.py
```

Do not expose the manager publicly without authentication. Anyone who can access it may be able to initiate large downloads into your environment.

## Configuration

### Maximum Civitai download size

The default limit is 12 GB:

```bash
FOOOCUS_CIVITAI_MAX_GB=20 python civitai_manager.py
```

### Standard model folders

```text
models/checkpoints/   SDXL checkpoints
models/loras/         LoRA files
models/vae/           VAE files
models/embeddings/    Textual inversion embeddings
```

Renewed Fooocus preserves the original Fooocus folder layout so existing compatible model libraries can be reused.

## Core Fooocus capabilities retained

Renewed Fooocus continues to provide the major workflows from the original project:

- Text-to-image generation
- Prompt expansion and style presets
- SDXL checkpoint and refiner selection
- Multiple LoRAs with adjustable weights
- Image variation and upscaling
- Inpainting and outpainting
- Image Prompt, FaceSwap, PyraCanny, and CPDS controls
- Image description tools
- Metadata import and export
- Advanced sampler, scheduler, guidance, and debugging controls

## Compatibility notes

Renewed Fooocus is primarily built around the **SDXL model family**. The Civitai Manager defaults to assets likely to work with this pipeline.

The manager can display other architectures in download-only mode, but downloading a file does not guarantee that Renewed Fooocus can load or generate with it. Flux, SD 1.5-only, Stable Cascade, and other architectures may require different software or pipelines.

Always review a model's license and usage terms on Civitai before using or redistributing it.

## Troubleshooting

### A model does not appear in the dropdown

Open **Advanced → Models** and click **Refresh All Files**. Confirm the file was installed into `models/checkpoints` or `models/loras` and has a supported extension.

### Colab disconnects or runs out of memory

- Use a smaller checkpoint or resolution.
- Reduce image count.
- Disable memory-intensive image-prompt or enhancement workflows.
- Restart the runtime after a large model switch.
- Keep only the models you actively use in mounted Drive folders.

### Corrupted or incomplete model

Delete both the model and any leftover `.part` file, then download it again. Interrupted downloads and exhausted storage are common causes.

### Civitai download requires an account

Create a Civitai API token in your Civitai account settings and paste it into the optional token field. The manager uses it for the current process and does not intentionally save it.

### Public Gradio link does not appear

Inspect the relevant notebook log output, confirm outbound connections are available, and rerun the manager or launch cell. Gradio share URLs are temporary and expire when the process or Colab runtime stops.

## Project scope

Renewed Fooocus focuses on:

- Maintaining a simple Fooocus-style SDXL experience
- Improving cloud-notebook deployment
- Making compatible model discovery and installation easier
- Preserving compatibility with existing Fooocus presets and model folders
- Adding practical quality-of-life improvements without replacing the core generation workflow

## Upstream and licensing

Renewed Fooocus is derived from [Fooocus by lllyasviel and its contributors](https://github.com/lllyasviel/Fooocus).

The upstream project introduced and maintained the core user interface, generation pipeline, prompt processing, inpainting, image-prompt systems, presets, and supporting modules on which this fork is based.

This repository retains the upstream license. See [`LICENSE`](LICENSE) and the repository history for full copyright and contributor information.

- Original project: [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus)
- Renewed fork: [JaeTheOP/fooocus-clone](https://github.com/JaeTheOP/fooocus-clone)
- Civitai integration details: [`CIVITAI_MANAGER.md`](CIVITAI_MANAGER.md)

## Disclaimer

You are responsible for the models you download, the licenses attached to them, the prompts and source images you provide, and the outputs you create. Do not use the software to violate applicable law, privacy rights, intellectual-property rights, platform rules, or model licenses.
