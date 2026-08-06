import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from boxing_vision.ui import build_app

if __name__ == "__main__":
    demo = build_app()
    demo.queue(default_concurrency_limit=1, max_size=4).launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=False,
    )
