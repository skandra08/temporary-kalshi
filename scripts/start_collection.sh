#!/usr/bin/env bash
# Start (or restart) the order-book collector and the archive loop in the background, tracked by pidfiles.
#   bash scripts/start_collection.sh [clean_repo_dir] [hours]
cd "$(dirname "$0")/.."
REPO="${1:-/home/user/kalshi-clean}"; HOURS="${2:-48}"
PIDDIR="${PIDDIR:-/tmp/collector_pids}"; mkdir -p "$PIDDIR"
for f in "$PIDDIR"/collector.pid "$PIDDIR"/archive.pid; do
  [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"
done
sleep 1
nohup python scripts/book_collector.py --all-mentions --fast-days 3 --max-markets 120 --every 60 --min-vol24 1000 --hours "$HOURS" > "${LOG:-/tmp/claude-0/s/book3.log}" 2>&1 &
echo $! > "$PIDDIR/collector.pid"
nohup bash scripts/archive_loop.sh "$REPO" > "${ARCHIVE_LOG:-/tmp/claude-0/s/archive.log}" 2>&1 &
echo $! > "$PIDDIR/archive.pid"
echo "started collector $(cat "$PIDDIR/collector.pid") and archive loop $(cat "$PIDDIR/archive.pid")"
