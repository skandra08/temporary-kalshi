#!/usr/bin/env bash
# Start (or restart) the order-book collector and the archive loop in the background.
#   bash scripts/start_collection.sh [clean_repo_dir] [hours]
cd "$(dirname "$0")/.."
REPO="${1:-/home/user/kalshi-clean}"; HOURS="${2:-48}"
for pat in "scripts/book_collector" "scripts/archive_loop"; do
  for pid in $(pgrep -f "$pat" || true); do [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null; done
done
sleep 1
nohup python scripts/book_collector.py --all-mentions --fast-days 3 --max-markets 120 --every 60 --min-vol24 1000 --hours "$HOURS" > /tmp/claude-0/s/book3.log 2>&1 &
nohup bash scripts/archive_loop.sh "$REPO" > /tmp/claude-0/s/archive.log 2>&1 &
echo started
