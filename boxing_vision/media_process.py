"""Small cross-platform process options for the local FFmpeg toolchain."""

from __future__ import annotations

import subprocess
import sys


def hidden_process_kwargs() -> dict[str, int]:
    """Suppress child console windows without changing non-Windows launches."""
    if sys.platform != "win32":
        return {}
    # The documented Win32 flag is absent from subprocess on non-Windows test
    # hosts; using its value also makes the platform branch testable there.
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


def stop_process(process: subprocess.Popen) -> None:
    """Reap a directly launched media process, escalating only after 3 seconds."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=3)
