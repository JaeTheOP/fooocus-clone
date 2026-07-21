import os

from modules.civitai_manager import launch_civitai_manager


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    host = os.getenv("FOOOCUS_CIVITAI_HOST", "127.0.0.1")
    port = int(os.getenv("FOOOCUS_CIVITAI_PORT", "7866"))
    share = _env_flag("FOOOCUS_CIVITAI_SHARE", False)

    username = os.getenv("FOOOCUS_CIVITAI_USERNAME", "").strip()
    password = os.getenv("FOOOCUS_CIVITAI_PASSWORD", "")
    auth = (username, password) if username and password else None

    launch_civitai_manager(
        host=host,
        port=port,
        share=share,
        auth=auth,
    )
