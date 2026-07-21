<div align="center">

# Renewed Fooocus

### A community continuation of Fooocus with Civitai model management

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaeTheOP/fooocus-clone/blob/agent/civitai-model-manager/colab.ipynb)
[![Based on Fooocus](https://img.shields.io/badge/based%20on-Fooocus-5b6cff)](https://github.com/lllyasviel/Fooocus)
[![License](https://img.shields.io/github/license/JaeTheOP/fooocus-clone)](LICENSE)

**The familiar Fooocus workflow, renewed with easier model discovery and installation.**

</div>

---

## What is Renewed Fooocus?

**Renewed Fooocus** is an independent community continuation of the original [Fooocus](https://github.com/lllyasviel/Fooocus) SDXL image-generation interface.

It preserves the core Fooocus experience while adding a companion Civitai manager for searching, selecting, and installing checkpoints and LoRAs.

> **Independent fork:** Renewed Fooocus is not the official Fooocus project and is not affiliated with Civitai. Full upstream attribution appears below.

## Main features

- Simple SDXL text-to-image generation
- Prompt expansion and style presets
- Checkpoint, refiner, and LoRA selection
- Image variation and upscaling
- Inpainting and outpainting
- Image Prompt, FaceSwap, PyraCanny, and CPDS controls
- Metadata import and export
- Companion Civitai checkpoint and LoRA manager
- Automatic installation into the configured Fooocus model folders
- `.safetensors` enforcement and model-download validation
- Google Colab support using the original three-cell notebook workflow

## Google Colab

Open the notebook:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaeTheOP/fooocus-clone/blob/agent/civitai-model-manager/colab.ipynb)

The notebook has been restored to the original simple three-cell layout.

### Run it

1. Open the notebook.
2. Choose **Runtime → Change runtime type**.
3. Select **T4 GPU** or another available NVIDIA GPU.
4. Run the first cell to clone the repository and install dependencies.
5. Optionally edit and run the second cell to download a custom Civitai checkpoint or LoRA.
6. Run the third cell to launch Renewed Fooocus.
7. Open the temporary `gradio.live` URL printed in the output.

Keep the final cell running while using the interface. Stopping it closes the public Gradio session.

### Original notebook cells

#### Cell 1 — Setup

The first cell:

- Installs `pygit2`
- Clones `JaeTheOP/fooocus-clone`
- Installs PyTorch 2.3.1 and Torchvision 0.18.1 with CUDA 12.1 wheels
- Installs the pinned project requirements

#### Cell 2 — Optional custom models

The second cell contains commented examples for downloading models directly from Civitai:

```python
# Checkpoint
# !wget -c --content-disposition -P models/checkpoints "https://civitai.com/api/download/models/XXXXX"

# LoRA
# !wget -c --content-disposition -P models/loras "https://civitai.com/api/download/models/YYYYY"
```

Replace `XXXXX` or `YYYYY` with the desired Civitai model-version ID and remove the leading `#` before running the command.

#### Cell 3 — Launch

The final cell launches the `photoreal_civitai` preset:

```bash
python entry_with_update.py --share --always-high-vram --preset photoreal_civitai
```

### Colab storage warning

The original notebook stores files in Colab's temporary `/content` storage. Models and generated images can disappear when the runtime resets or is deleted. Download important outputs before ending the session.

## Local installation

### Requirements

- Python 3.10
- Git
- An NVIDIA GPU is recommended
- At least 4 GB VRAM for basic generation
- Enough free storage for SDXL checkpoints and outputs

### Linux or macOS

```bash
git clone --branch agent/civitai-model-manager https://github.com/JaeTheOP/fooocus-clone.git renewed-fooocus
cd renewed-fooocus
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
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
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_versions.txt
python entry_with_update.py
```

The correct PyTorch command can vary by GPU and driver. Use the official PyTorch installation selector when CUDA 12.1 is not appropriate for your computer.

## Civitai Manager

The companion manager provides a searchable interface for installing checkpoints and LoRAs.

Run it from the repository root:

```bash
python civitai_manager.py
```

Default address:

```text
http://127.0.0.1:7866
```

The manager can:

- Search Civitai for checkpoints and LoRAs
- Sort by downloads, rating, or newest release
- Select a specific model version
- Filter for likely Fooocus-compatible SDXL assets
- Display architecture, size, compatibility, and trigger words
- Install into the correct checkpoint or LoRA directory
- Refresh the local model index after installation
- Use an optional Civitai API token for restricted downloads

### Public manager link

To create a temporary authenticated Gradio link:

```bash
FOOOCUS_CIVITAI_SHARE=1 \
FOOOCUS_CIVITAI_USERNAME=renewed \
FOOOCUS_CIVITAI_PASSWORD='replace-this-password' \
python civitai_manager.py
```

Do not expose the manager publicly without authentication. Anyone with access could initiate large downloads into the environment.

## Download protections

The Civitai downloader:

- Accepts `.safetensors` checkpoint and LoRA files
- Rejects failed Civitai virus or pickle scan results
- Validates download hosts, filenames, and destination paths
- Uses temporary `.part` files and atomic replacement
- Enforces a configurable maximum download size
- Verifies SHA-256 when Civitai supplies a complete hash
- Writes a `.civitai.json` metadata sidecar
- Keeps the optional API token in memory

The default maximum download size is 12 GB. For local use, change it with:

```bash
FOOOCUS_CIVITAI_MAX_GB=20 python civitai_manager.py
```

## Model folders

```text
models/checkpoints/   SDXL checkpoints
models/loras/         LoRA files
models/vae/           VAE files
models/embeddings/    Textual inversion embeddings
```

Renewed Fooocus preserves the original Fooocus folder layout so existing compatible model libraries can be reused.

## Compatibility

Renewed Fooocus is primarily designed around the **SDXL model family**. Downloading an incompatible architecture does not make it usable by Fooocus.

Flux, SD 1.5, Stable Cascade, and other architectures may require different software or pipelines. Review each model's license and usage terms before using or redistributing it.

## Troubleshooting

### No GPU appears in Colab

Choose **Runtime → Change runtime type**, select **T4 GPU**, save the setting, and rerun the notebook from the first cell.

### A model does not appear

Open **Advanced → Models** and click **Refresh All Files**. Confirm that the model was placed in `models/checkpoints` or `models/loras` and uses a supported file format.

### A download is incomplete or corrupted

Delete the model and any remaining `.part` file, then download it again. Interrupted sessions and exhausted temporary storage are common causes in Colab.

### A public Gradio URL is not created

Read the final notebook output for the underlying error. Gradio share URLs are temporary and depend on outbound network access.

## Validation

The repository's validation workflow:

- Compiles the Civitai manager Python modules
- Confirms the original three-cell Colab notebook is valid notebook JSON
- Runs the Civitai client unit tests

A true end-to-end generation test still requires a compatible NVIDIA GPU runtime.

## Upstream and licensing

Renewed Fooocus is derived from [Fooocus by lllyasviel and its contributors](https://github.com/lllyasviel/Fooocus).

The upstream project created the core generation pipeline, prompt processing, inpainting, image-prompt systems, presets, and user interface on which this fork is based.

This repository retains the upstream license. See [`LICENSE`](LICENSE) and the repository history for copyright and contributor details.

- Original project: [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus)
- Renewed fork: [JaeTheOP/fooocus-clone](https://github.com/JaeTheOP/fooocus-clone)
- Civitai integration documentation: [`CIVITAI_MANAGER.md`](CIVITAI_MANAGER.md)

## Disclaimer

You are responsible for the models you download, their licenses, the prompts and source images you provide, and the outputs you create. Do not use the software to violate applicable law, privacy rights, intellectual-property rights, platform rules, or model licenses.
