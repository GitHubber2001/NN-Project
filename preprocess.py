"""
Preprocessing, train/val/test split and windowing for one-step prediction.

Key design choices (to justify in the report):
  * SPLIT BY TRAJECTORY (never by time step) to avoid leakage between
    train and test; test trajectories come from unseen initial conditions.
  * Squashing: z = tanh(x / c_k), c_k = scale * std_k (fitted on TRAIN only).
    This compresses rare large excursions (a body being flung out) and is
    invertible: x = c_k * arctanh(z).  Positions and velocities get separate
    scales.
  * Network time step = `stride` * dt_sample: sub-sample the stored series so
    the RNN predicts state(t + stride*dt) from state(t).
"""
import numpy as np


# ------------------------------------------------------------------ loading
def load_dataset(path="data/threebody_raw.npz"):
    """Returns (list of (len_i,12) arrays at the stored step dt, dt, E0)."""
    d = np.load(path)
    arr, lens = d["trajs"], d["lengths"]
    return [arr[i, :lens[i]] for i in range(len(arr))], float(d["dt"]), float(d["E0"])


# ------------------------------------------------------------------ split
def split_trajectories(trajs, frac=(0.7, 0.15, 0.15), seed=0):
    idx = np.random.default_rng(seed).permutation(len(trajs))
    n_tr = int(frac[0] * len(trajs)); n_va = int(frac[1] * len(trajs))
    pick = lambda ii: [trajs[i] for i in ii]
    return pick(idx[:n_tr]), pick(idx[n_tr:n_tr + n_va]), pick(idx[n_tr + n_va:])


def subsample(trajs, stride):
    """Network step = stride * dt_sample."""
    return [t[::stride] for t in trajs]


# ------------------------------------------------------------ normalisation
class Squasher:
    """z = tanh(x / c); separate scale for positions (cols 0-5) and velocities (6-11)."""
    def __init__(self, scale=2.0):
        self.scale = scale

    def fit(self, trajs):
        X = np.concatenate(trajs, axis=0)
        self.c = np.empty(12)
        self.c[:6] = self.scale * X[:, :6].std()
        self.c[6:] = self.scale * X[:, 6:].std()
        return self

    def transform(self, x):
        return np.tanh(x / self.c)

    def inverse(self, z):
        return self.c * np.arctanh(np.clip(z, -0.999999, 0.999999))


# ---------------------------------------------------------------- windows
def make_windows(trajs, seq_len=50, hop=None):
    """Cut trajectories into windows for one-step-ahead training.
    Returns X, Y of shape (N, seq_len, 12) with Y[:, k] = X[:, k+1]
    (i.e. the target is the input shifted by one network step)."""
    hop = hop or seq_len // 2
    X, Y = [], []
    for t in trajs:
        for a in range(0, len(t) - seq_len - 1 + 1, hop):
            X.append(t[a:a + seq_len]); Y.append(t[a + 1:a + seq_len + 1])
    return np.array(X), np.array(Y)


def make_pairs(trajs):
    """(state_t, state_{t+1}) pairs for a plain MLP / the kNN-or-linear baseline."""
    X = np.concatenate([t[:-1] for t in trajs]); Y = np.concatenate([t[1:] for t in trajs])
    return X, Y


# ------------------------------------------------------- optional torch glue
def to_torch_loaders(X, Y, batch_size=64, shuffle=True):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
