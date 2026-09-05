"""PyInstaller/native entrypoint; no web browser window is opened."""

from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    from boxing_vision.desktop import main

    raise SystemExit(main())
