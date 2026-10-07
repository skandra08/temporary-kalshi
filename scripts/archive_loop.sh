#!/usr/bin/env bash
# Every 30 min: gzip the collected order-book files and push them to the clean repo so a machine reset cannot lose them.
#   bash scripts/archive_loop.sh /path/to/clean/repo
set -u
REPO="${1:-/home/user/kalshi-clean}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
while true; do
  mkdir -p "$REPO/data_live"
  for f in "$SRC"/data_live/book_*.jsonl; do
    [ -e "$f" ] && gzip -c "$f" > "$REPO/data_live/$(basename "$f").gz"
  done
  cd "$REPO" || exit 1
  git add data_live
  if ! git diff --cached --quiet; then
    git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit -q -m "data: order-book snapshots $(date -u +%Y%m%d_%H%M)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TvpQbLHrHjyVNyWtaLKA3L"
    git pull -q --rebase origin main 2>/dev/null; git push -q origin main 2>/dev/null || echo "push failed $(date -u)"
  fi
  sleep 1800
done
