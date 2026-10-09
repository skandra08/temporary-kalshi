import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdskill.fetch import fetch_all, list_resolved_binary
root = Path("data/manifold"); root.mkdir(parents=True, exist_ok=True)
ms = list_resolved_binary(root / "markets.json", pages=int(sys.argv[1]) if len(sys.argv) > 1 else 80)
print(len(ms), "markets", flush=True)
fetch_all(ms, root / "bets")
print("DONE", flush=True)
