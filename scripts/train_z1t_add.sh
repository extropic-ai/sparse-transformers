#!/usr/bin/env bash
# Train Z1T with causal AFT-conv attention on the addition dataset.
#
# run from the repo root regardless of where this is invoked from
cd "$(dirname "$0")/.." || exit 1
# prefer the repo venv when it exists, so this works without activating it
PY=python
[ -x .venv/bin/python ] && PY=.venv/bin/python
"$PY" -m z1t.train --config configs/z1t_addition_aft_conv.yaml "$@"
