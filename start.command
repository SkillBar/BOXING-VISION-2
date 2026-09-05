#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
export GRADIO_ANALYTICS_ENABLED="False"

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "Boxing Vision: нужен FFmpeg. Установите: brew install ffmpeg"
  read -r "?Нажмите Enter, чтобы закрыть окно..."
  exit 1
fi

if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "libx264"; then
  echo "Boxing Vision: установленный FFmpeg не содержит encoder libx264."
  echo "Установите полную сборку: brew reinstall ffmpeg"
  read -r "?Нажмите Enter, чтобы закрыть окно..."
  exit 1
fi

PYPROJECT_HASH="$(shasum -a 256 pyproject.toml | awk '{print $1}')"
DEPENDENCY_STAMP=".venv/.boxing-vision-${PYPROJECT_HASH}"
if [[ ! -f "$DEPENDENCY_STAMP" ]]; then
  # Avoid build isolation so a prepared Mac can relaunch fully offline. A new
  # environment still resolves runtime wheels once, then the hash stamp skips
  # pip on subsequent launches.
  .venv/bin/python -m pip install --disable-pip-version-check --no-build-isolation -e .
  find .venv -maxdepth 1 -name '.boxing-vision-*' ! -name ".boxing-vision-${PYPROJECT_HASH}" -delete
  : > "$DEPENDENCY_STAMP"
fi

exec .venv/bin/python app.py
