"""Generate the dataset.   python make_data.py [--force] [--out data/threebody_raw.npz]

WARNING: the three-body problem is chaotic, so re-generating on another machine /
SciPy version gives slightly different trajectories. The team uses ONE shared file
(data/threebody_raw.npz, committed to the repo). This script therefore refuses to
overwrite an existing file unless --force is given.
"""
import os, sys, numpy as np
from data_gen import *

N_TRAJ, T, DT, E0, SEED = 300, 20.0, 0.01, -0.5, 0
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "data/threebody_raw.npz"
if os.path.exists(out) and "--force" not in sys.argv:
    sys.exit(f"{out} already exists - use the shared file (or pass --force to overwrite).")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

trajs = generate_dataset(N_TRAJ, T, DT, E0, SEED)
arr, lens = pad_stack(trajs)
np.savez_compressed(out, trajs=arr, lengths=lens, dt=DT, E0=E0, T=T)
print("saved", out, arr.shape, "mean length", lens.mean())
