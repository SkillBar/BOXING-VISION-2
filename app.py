import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from boxing_vision.ui import build_app


def _server_port() -> int:
    raw = os.environ.get("BOXING_VISION_PORT", "7860")
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit("BOXING_VISION_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise SystemExit("BOXING_VISION_PORT must be between 1 and 65535")
    return port

if __name__ == "__main__":
    demo = build_app()
    project_root = Path(__file__).resolve().parent
    allowed_paths = [str(project_root / "boxing_vision" / "static"), str(project_root / "runs")]
    prepared_demo = os.environ.get("BOXING_VISION_DEMO_RUN", "")
    if prepared_demo and (Path(prepared_demo) / "demo-manifest.json").is_file():
        from boxing_vision.demo_bundle import verify_demo
        verify_demo(Path(prepared_demo))
        allowed_paths.append(str(Path(prepared_demo).resolve()))
    demo.queue(default_concurrency_limit=1, max_size=4).launch(
        server_name="127.0.0.1",
        server_port=_server_port(),
        show_error=True,
        inbrowser=False,
        allowed_paths=allowed_paths,
    )
