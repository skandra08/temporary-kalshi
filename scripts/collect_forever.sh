#!/usr/bin/env bash
# Local forward-data collector: no cloud credits needed. Samples the live Kalshi BTC ladder,
# spot and Deribit smile every 5 min, and every ~hour commits + pushes the new data file.
#   bash scripts/collect_forever.sh            # leave running in a terminal / tmux
# Stop with Ctrl-C. Needs: pip install -e .   and git credentials already set up.
set -u
cd "$(dirname "$0")/.."
while true; do
  python -m digitaledge.collect --minutes 55 --every 300
  git add data_live
  if ! git diff --cached --quiet; then
    git commit -q -m "data: forward snapshots $(date -u +%Y%m%d_%H%M)"
    git pull -q --rebase origin main && git push -q origin main || echo "push failed; will retry next cycle"
  fi
done
