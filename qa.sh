#!/usr/bin/env bash
set -e

NOTEBOOKS=("main.ipynb" "echo-next.ipynb")
PY_FILES=("utils.py")

nbqa black "${NOTEBOOKS[@]}"
black "${PY_FILES[@]}"

nbqa isort "${NOTEBOOKS[@]}"
isort "${PY_FILES[@]}"

nbqa flake8 --ignore=E501,E226,E402,W503 "${NOTEBOOKS[@]}"
flake8 --ignore=E501,E226,E402,W503 "${PY_FILES[@]}"
