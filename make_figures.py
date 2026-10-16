"""Report figures from data/threebody_raw.npz.  python make_figures.py"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_gen import *
from preprocess import *

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3})
d = np.load("data/threebody_raw.npz"); arr, lens, DT, E0 = d["trajs"], d["lengths"], float(d["dt"]), float(d["E0"])
trajs = [arr[i, :lens[i]] for i in range(len(arr))]
cols = ["tab:red", "tab:green", "tab:blue"]

# Fig 1: example trajectories
fig, axs = plt.subplots(2, 2, figsize=(7, 6))
for k, (ax, t) in enumerate(zip(axs.ravel(), trajs[:4])):
    p, _ = split_state(t)
    for b in range(3):
        ax.plot(p[:, b, 0], p[:, b, 1], color=cols[b], lw=0.8, label=f"body {b+1}")
        ax.plot(*p[0, b], "o", color=cols[b], ms=6)
    ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"trajectory {k+1}, $t\\in[0,{(len(t)-1)*DT:.1f}]$", fontsize=9)
axs[0, 0].legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig("figs/example_trajectories.pdf"); plt.close()

# Fig 2: effect of squashing (train data only)
tr, va, te = split_trajectories(trajs); sq = Squasher().fit(tr)
X = np.concatenate(tr); Z = sq.transform(X)
fig, axs = plt.subplots(1, 2, figsize=(7, 2.8), sharey=False)
for ax, A, ttl in zip(axs, (X, Z), ("raw", "after $\\tanh(x/c_k)$")):
    ax.hist(A[:, :6].ravel(), 80, alpha=.6, label="positions")
    ax.hist(A[:, 6:].ravel(), 80, alpha=.6, label="velocities")
    ax.set_title(ttl, fontsize=9); ax.set_xlabel("value"); ax.set_ylabel("count")
axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figs/preprocessing.pdf"); plt.close()

# Fig 3: energy conservation of the reference solver
fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
drift = np.array([np.abs(total_energy(t) - E0).max() for t in trajs])
axs[0].hist(drift, np.logspace(np.log10(drift.min()), np.log10(drift.max()), 30))
axs[0].set_xscale("log"); axs[0].set_xlabel("$\\max_t |E(t)-E_0|$"); axs[0].set_ylabel("# trajectories")
for t in trajs[:5]:
    axs[1].semilogy(np.arange(len(t)) * DT, np.abs(total_energy(t) - E0) + 1e-16, lw=.8)
axs[1].set_xlabel("time $t$"); axs[1].set_ylabel("$|E(t)-E_0|$")
plt.tight_layout(); plt.savefig("figs/energy_check.pdf"); plt.close()

# Table: how the network time step changes one-step difficulty (validation set)
print("stride  dt_net  copy-last-MSE  linear-MSE  ratio")
for s in (1, 2, 5, 10, 20):
    a, b = [subsample(x, s) for x in (tr, va)]
    sq = Squasher().fit(a); Xt, Yt = make_pairs([sq.transform(t) for t in a]); Xv, Yv = make_pairs([sq.transform(t) for t in b])
    A = lambda M: np.c_[M, np.ones(len(M))]
    W = np.linalg.lstsq(A(Xt), Yt, rcond=None)[0]
    c, l = ((Xv - Yv) ** 2).mean(), ((A(Xv) @ W - Yv) ** 2).mean()
    print(f"{s:5d}  {s*DT:6.2f}  {c:.3e}  {l:.3e}  {l/c:.3f}")
print("drift median/worst", np.median(drift), drift.max())
print("N samples train", sum(len(t) for t in tr), "val", sum(len(t) for t in va), "test", sum(len(t) for t in te))
print("scales c (pos, vel):", sq.c[0], sq.c[6])
