#!/usr/bin/env bash
set -euo pipefail

echo "=== Python / pip environment diagnostic ==="

echo "[which python]" && which python || true
echo "[python --version]" && python --version || true

echo "[platform]" && python - <<'PY'
import platform, sys
print('sys.executable:', sys.executable)
print('python:', sys.version.split()[0])
print('system:', platform.system())
print('machine:', platform.machine())
print('platform:', platform.platform())
PY

echo "[python -m pip --version]" && python -m pip --version || true

echo "[pip list (first 30)]" && python -m pip list | head -n 30 || true

echo "\nNOTE: If your shell prompt shows both '(base)' and '(venv)', you're mixing conda + venv."
echo "Prefer ONE: conda env OR venv, not both."
