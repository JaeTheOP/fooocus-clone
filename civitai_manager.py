import os

from modules.civitai_manager import launch_civitai_manager


if __name__ == "__main__":
    host = os.getenv("FOOOCUS_CIVITAI_HOST", "127.0.0.1")
    port = int(os.getenv("FOOOCUS_CIVITAI_PORT", "7866"))
    launch_civitai_manager(host=host, port=port)
