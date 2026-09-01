#!/usr/bin/env bash
# Train Z1T on char-level Shakespeare at tiny scale.
#
# run from the repo root regardless of where this is invoked from
cd "$(dirname "$0")/.." || exit 1
# prefer the repo venv when it exists, so this works without activating it
PY=python
[ -x .venv/bin/python ] && PY=.venv/bin/python
# the st / z1t packages live under research/, so put it on the import path
export PYTHONPATH="research${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m z1t.train --config configs/z1t_tiny.yaml "$@"
