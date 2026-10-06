#!/usr/bin/env bash
# One-time local setup: virtualenv, dependencies, tests.   bash scripts/setup_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[dev]"
pytest -q
echo
echo "Setup OK. Next:"
echo "  source .venv/bin/activate"
echo "  export EDGAR_UA=\"kalshi-edge-research you@example.com\"   # SEC asks for a contact email"
echo "  python scripts/earnings_experiment.py        # earnings-call text model (first run downloads ~1 GB of filings)"
echo "  python scripts/mentions_sports_experiment.py # announcer-mention models"
