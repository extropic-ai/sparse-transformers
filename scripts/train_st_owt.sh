#!/usr/bin/env bash
# Train the st sparse transformer (windowed softmax attention) on OpenWebText.
# Expects <data_path>/{train.npy,validation.npy} — see configs/st_owt.yaml.
#
# run from the repo root regardless of where this is invoked from
cd "$(dirname "$0")/.." || exit 1
# prefer the repo venv when it exists, so this works without activating it
PY=python
[ -x .venv/bin/python ] && PY=.venv/bin/python
# the st / z1t packages live under research/, so put it on the import path
export PYTHONPATH="research${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m experiments.text_scaling --config configs/st_owt.yaml "$@"
