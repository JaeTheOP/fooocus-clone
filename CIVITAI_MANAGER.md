# Renewed Fooocus Civitai Manager

Renewed Fooocus includes a standalone Gradio interface for searching and installing Civitai checkpoints and LoRAs.

The manager is intended primarily for local or server installations. The restored Google Colab notebook uses the original three-cell workflow and includes commented manual `wget` examples instead of starting this manager automatically.

## Start the manager

From the repository root:

```bash
python civitai_manager.py
```

Default address:

```text
http://127.0.0.1:7866
```

## Search and installation

The manager can:

- Search Civitai for checkpoints or LoRAs
- Sort by most downloaded, highest rated, or newest
- Select a specific model version
- Default to likely Fooocus-compatible SDXL-family assets
- Display the base architecture, file size, compatibility, and trigger words
- Install checkpoints into the configured checkpoint folder
- Install LoRAs into the configured LoRA folder
- Refresh Fooocus's internal model index after installation
- Use an optional Civitai API token for account-restricted downloads

After installing a model, open Renewed Fooocus and use **Advanced → Models → Refresh All Files** before selecting the new asset.

## Download protections

The downloader:

- Accepts `.safetensors` files
- Rejects failed Civitai virus or pickle scan results
- Validates filenames and destination paths
- Restricts model downloads to expected Civitai hosts
- Downloads to a temporary `.part` file before atomic replacement
- Enforces a maximum download size
- Verifies SHA-256 when Civitai provides a complete hash
- Writes a `.civitai.json` metadata sidecar
- Does not intentionally write the optional API token to disk

The default maximum download size is 12 GB. Change it with:

```bash
FOOOCUS_CIVITAI_MAX_GB=20 python civitai_manager.py
```

## Network access

Listen on all network interfaces:

```bash
FOOOCUS_CIVITAI_HOST=0.0.0.0 python civitai_manager.py
```

Change the port:

```bash
FOOOCUS_CIVITAI_PORT=7867 python civitai_manager.py
```

Create a temporary public Gradio link:

```bash
FOOOCUS_CIVITAI_SHARE=1 python civitai_manager.py
```

Protect a public link with authentication:

```bash
FOOOCUS_CIVITAI_SHARE=1 \
FOOOCUS_CIVITAI_USERNAME=renewed \
FOOOCUS_CIVITAI_PASSWORD='replace-this-password' \
python civitai_manager.py
```

Do not expose the manager publicly without authentication. Anyone with access may be able to start large model downloads into the configured environment.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `FOOOCUS_CIVITAI_HOST` | `127.0.0.1` | Gradio server host |
| `FOOOCUS_CIVITAI_PORT` | `7866` | Gradio server port |
| `FOOOCUS_CIVITAI_SHARE` | disabled | Creates a temporary Gradio share URL |
| `FOOOCUS_CIVITAI_USERNAME` | empty | Optional Gradio username |
| `FOOOCUS_CIVITAI_PASSWORD` | empty | Optional Gradio password |
| `FOOOCUS_CIVITAI_MAX_GB` | `12` | Maximum permitted model download size |

## Compatibility

Renewed Fooocus is primarily designed for the SDXL model family. The manager can expose additional architectures in download-only mode, but downloading an asset does not guarantee that Fooocus can load it.

Flux, SD 1.5, Stable Cascade, and other architectures may require different software or pipelines.

## Google Colab

The repository's `colab.ipynb` has been restored to its original three-cell structure:

1. Install dependencies and clone the repository.
2. Optionally download custom Civitai files using commented `wget` examples.
3. Launch Fooocus with the `photoreal_civitai` preset.

The original notebook does not automatically run the Civitai Manager, generate manager credentials, or mount Google Drive.

## Scope

This integration changes model discovery and installation only. It does not remove Fooocus content-safety controls or modify its generation pipeline.
