"""Verify that you have the official shared dataset.   python check_data.py"""

import hashlib

import numpy as np

from data_gen import total_energy
from preprocess import *

PATH = "data/threebody_raw.npz"
trajs, dt, E0 = load_dataset(PATH)
drift = np.array([np.abs(total_energy(t) - E0).max() for t in trajs])
tr, va, te = split_trajectories(trajs)
print("sha256 (first 16):", hashlib.sha256(open(PATH, "rb").read()).hexdigest()[:16])
print("trajectories:", len(trajs), " mean length:", np.mean([len(t) for t in trajs]))
print(f"energy drift: median {np.median(drift):.2e}, worst {drift.max():.2e}")
print(
    "samples train/val/test:", sum(map(len, tr)), sum(map(len, va)), sum(map(len, te))
)
print(
    "EXPECTED: 300 trajectories, mean length 1649.15, drift median 9.60e-10 / worst 6.67e-04,"
)
print("          samples 344443 / 75033 / 75269")
