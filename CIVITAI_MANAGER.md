# Fooocus Civitai Manager

This branch adds a standalone Civitai browser and installer for Fooocus checkpoints and LoRAs.

## Run

Install Fooocus normally, then start the manager from the repository root:

```bash
python civitai_manager.py
```

Open `http://127.0.0.1:7866` while Fooocus is running on its normal port.

Environment overrides:

```bash
FOOOCUS_CIVITAI_HOST=0.0.0.0 FOOOCUS_CIVITAI_PORT=7866 python civitai_manager.py
```

The maximum accepted download size defaults to 12 GB and can be changed with:

```bash
FOOOCUS_CIVITAI_MAX_GB=20 python civitai_manager.py
```

## Features

- Searches Civitai checkpoints and LoRAs.
- Selects a specific model version rather than only a model page.
- Defaults to SDXL-family assets compatible with Fooocus.
- Offers an all-architectures mode for downloading files for other tools.
- Installs checkpoints into the configured checkpoint path.
- Installs LoRAs into the configured LoRA path.
- Accepts only `.safetensors` files.
- Rejects files whose Civitai virus or pickle scan reports danger/error.
- Verifies SHA-256 when Civitai supplies a full SHA-256 hash.
- Uses temporary `.part` files and atomic replacement.
- Writes a sidecar `.civitai.json` metadata file.
- Keeps the optional Civitai API token in memory only.

After installation, click **Refresh All Files** in Fooocus's Models tab to refresh the visible model and LoRA dropdowns.

## Scope

This integration changes model discovery and installation only. It does not remove Fooocus content-safety controls or modify its generation pipeline.
