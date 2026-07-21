from __future__ import annotations

import html
import os
from typing import Any

import gradio as gr

from modules import config
from modules.civitai_client import CivitaiError, download_model, search_models


COMPATIBILITY_CHOICES = [
    "Renewed Fooocus-compatible (SDXL family)",
    "All architectures (download only)",
]


def _label(record: dict[str, Any]) -> str:
    size_gb = float(record.get("size_kb") or 0) / 1024 / 1024
    compatibility = "compatible" if record.get("fooocus_compatible") else "download only"
    return (
        f"{record['model_name']} — {record['version_name']} | {record['base_model']} | "
        f"{size_gb:.2f} GB | {compatibility} [version:{record['version_id']}]"
    )


def _search_ui(query, model_type, sort, period, compatibility, api_token):
    try:
        records = search_models(
            query=query,
            model_type=model_type,
            sort=sort,
            period=period,
            compatibility=compatibility,
            api_token=api_token,
        )
    except CivitaiError as exc:
        return gr.update(choices=[], value=None), [], f"**Search failed:** {html.escape(str(exc))}"

    labels = [_label(record) for record in records]
    status = f"Found **{len(labels)}** installable `.safetensors` version(s)."
    if not labels:
        status += " Try a broader search or the all-architectures filter."
    return gr.update(choices=labels, value=labels[0] if labels else None), records, status


def _download_ui(selection, records, api_token, overwrite):
    if not selection:
        return "**Select a model version first.**"
    record = next((item for item in (records or []) if _label(item) == selection), None)
    if record is None:
        return "**The search result expired. Run the search again.**"

    destination = config.paths_checkpoints[0] if record["model_type"] == "Checkpoint" else config.paths_loras[0]
    try:
        result = download_model(
            record=record,
            destination_dir=destination,
            api_token=api_token,
            overwrite=bool(overwrite),
        )
        config.update_files()
    except CivitaiError as exc:
        return f"**Install failed:** {html.escape(str(exc))}"

    size_gb = result["bytes"] / 1024 ** 3
    trained_words = ", ".join(record.get("trained_words") or []) or "None listed"
    return (
        f"**Installed:** `{html.escape(os.path.basename(result['path']))}` ({size_gb:.2f} GB)  \n"
        f"**Type:** {record['model_type']}  \n"
        f"**Base architecture:** {html.escape(record['base_model'])}  \n"
        f"**Trigger words:** {html.escape(trained_words)}  \n"
        f"**SHA-256:** `{result['sha256']}`  \n\n"
        "Renewed Fooocus's file index is refreshed. In the main Renewed Fooocus window, click "
        "**Refresh All Files** to update the visible dropdown choices."
    )


def build_civitai_manager() -> gr.Blocks:
    with gr.Blocks(title="Renewed Fooocus Civitai Manager") as app:
        gr.Markdown(
            "# Renewed Fooocus Civitai Manager\n"
            "Search and install Civitai checkpoints or LoRAs without manually copying URLs. "
            "The manager accepts only `.safetensors`, rejects failed malware scans, verifies SHA-256 when provided, "
            "and does not save your API token."
        )
        records_state = gr.State([])

        with gr.Row():
            query = gr.Textbox(label="Search", placeholder="Model or LoRA name")
            model_type = gr.Radio(["Checkpoint", "LORA"], value="Checkpoint", label="Asset type")
        with gr.Row():
            sort = gr.Dropdown(
                ["Most Downloaded", "Highest Rated", "Newest"], value="Most Downloaded", label="Sort"
            )
            period = gr.Dropdown(["AllTime", "Year", "Month", "Week", "Day"], value="AllTime", label="Period")
            compatibility = gr.Dropdown(
                COMPATIBILITY_CHOICES, value=COMPATIBILITY_CHOICES[0], label="Compatibility"
            )
        api_token = gr.Textbox(
            label="Civitai API token (optional)",
            type="password",
            placeholder="Needed for gated/account-only downloads",
        )
        search_button = gr.Button("Search Civitai", variant="primary")
        results = gr.Dropdown(label="Model version", choices=[])
        overwrite = gr.Checkbox(label="Overwrite an existing file with the same name", value=False)
        install_button = gr.Button("Install selected version", variant="primary")
        status = gr.Markdown("Ready.")

        search_button.click(
            _search_ui,
            inputs=[query, model_type, sort, period, compatibility, api_token],
            outputs=[results, records_state, status],
            queue=True,
        )
        query.submit(
            _search_ui,
            inputs=[query, model_type, sort, period, compatibility, api_token],
            outputs=[results, records_state, status],
            queue=True,
        )
        install_button.click(
            _download_ui,
            inputs=[results, records_state, api_token, overwrite],
            outputs=status,
            queue=True,
        )

    return app.queue(concurrency_count=1)


def launch_civitai_manager(
    host: str = "127.0.0.1",
    port: int = 7866,
    share: bool = False,
    auth: tuple[str, str] | None = None,
) -> None:
    app = build_civitai_manager()
    launch_result = app.launch(
        server_name=host,
        server_port=port,
        share=share,
        auth=auth,
        prevent_thread_lock=False,
        show_error=True,
        quiet=False,
    )
    local_url = launch_result[1] if isinstance(launch_result, tuple) and len(launch_result) > 1 else None
    share_url = launch_result[2] if isinstance(launch_result, tuple) and len(launch_result) > 2 else None
    if local_url:
        print(f"Renewed Fooocus Civitai Manager local URL: {local_url}")
    if share_url:
        print(f"Renewed Fooocus Civitai Manager public URL: {share_url}")
