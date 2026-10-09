import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from attention.data import matrix, top_articles
arts = top_articles(n=int(sys.argv[1]) if len(sys.argv) > 1 else 700)
print(len(arts),'articles',flush=True)
Y, names, idx = matrix(arts)
np.savez("data/wiki_matrix.npz", Y=Y, names=np.array(names), idx=np.array(idx.strftime("%Y-%m-%d")))
print("DONE", Y.shape, flush=True)
