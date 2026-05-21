"""Entry point for `docchat-api` CLI command — starts uvicorn server."""

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("DOCCHAT_HOST", "0.0.0.0")
    # Railway (và các Cloud PaaS khác) tự động cấp phát cổng qua biến PORT
    port_str = os.environ.get("PORT") or os.environ.get("DOCCHAT_PORT", "8000")
    port = int(port_str)
    reload = os.environ.get("DOCCHAT_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "docchat.api:app",
        host=host,
        port=port,
        reload=reload,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
