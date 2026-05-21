"""Entry point for `docchat-api` CLI command — starts uvicorn server."""

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("DOCCHAT_HOST", "0.0.0.0")
    port = int(os.environ.get("DOCCHAT_PORT", "8000"))
    reload = os.environ.get("DOCCHAT_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "docchat.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
