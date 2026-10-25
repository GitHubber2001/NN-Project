"""
Data generation for the 2D three-body problem (equal masses, G = 1).

State layout (12-dim), used everywhere in the project:
    [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]

Pipeline:
  1. sample random initial conditions in the centre-of-mass frame,
  2. rescale velocities so that every trajectory has the SAME total energy E0
     (the project brief asks for this: the attractor geometry depends on E),
  3. integrate with a high-accuracy adaptive solver (DOP853, tight tolerances),
  4. cut each trajectory at the first moment a body escapes or two bodies get
     extremely close (those events are numerically unreliable and have a huge
     dynamic range, which neural networks dislike),
  5. save everything to one .npz file.
"""

import numpy as np
from scipy.integrate import solve_ivp

G = 1.0
MASSES = np.array([1.0, 1.0, 1.0])


# ---------------------------------------------------------------- physics
def split_state(s):
    """(…,12) -> pos (…,3,2), vel (…,3,2)"""
    s = np.asarray(s)
    pos = s[..., :6].reshape(s.shape[:-1] + (3, 2))
    vel = s[..., 6:].reshape(s.shape[:-1] + (3, 2))
    return pos, vel


def join_state(pos, vel):
    return np.concatenate(
        [pos.reshape(pos.shape[:-2] + (6,)), vel.reshape(vel.shape[:-2] + (6,))],
        axis=-1,
    )


def accelerations(pos, m=MASSES):
    """pos (3,2) -> acceleration (3,2) from Newtonian gravity."""
    d = pos[None, :, :] - pos[:, None, :]  # d[i,j] = r_j - r_i
    r3 = (d**2).sum(-1) ** 1.5
    np.fill_diagonal(r3, 1.0)  # avoid 0/0 on the diagonal
    w = G * m[None, :] / r3
    np.fill_diagonal(w, 0.0)
    return (w[:, :, None] * d).sum(1)


def rhs(t, s):
    pos, vel = s[:6].reshape(3, 2), s[6:].reshape(3, 2)
    return np.concatenate([vel.ravel(), accelerations(pos).ravel()])


def total_energy(s, m=MASSES):
    """Total energy for state(s) of shape (…,12)."""
    pos, vel = split_state(s)
    kin = 0.5 * (m * (vel**2).sum(-1)).sum(-1)
    pot = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            pot = pot - G * m[i] * m[j] / np.linalg.norm(
                pos[..., i, :] - pos[..., j, :], axis=-1
            )
    return kin + pot


def min_pair_distance(s):
    pos, _ = split_state(s)
    ds = [
        np.linalg.norm(pos[..., i, :] - pos[..., j, :], axis=-1)
        for i, j in ((0, 1), (0, 2), (1, 2))
    ]
    return np.min(ds, axis=0)


def max_radius(s):
    pos, _ = split_state(s)
    return np.linalg.norm(pos, axis=-1).max(-1)


# ---------------------------------------------------------- initial conditions
def sample_initial_condition(
    rng, E0=-0.5, pos_scale=1.0, r_min_init=0.3, max_tries=1000
):
    """Random planar IC in the COM frame with total energy exactly E0.

    Positions ~ N(0, pos_scale^2), velocity directions random; the velocity
    magnitude is rescaled by a single factor so that KE = E0 - U.
    Requires E0 < 0 (bound system) so that ICs are not too violent.
    """
    for _ in range(max_tries):
        pos = rng.normal(0.0, pos_scale, size=(3, 2))
        pos -= pos.mean(0)  # COM at origin (equal masses)
        if min_pair_distance(join_state(pos, np.zeros((3, 2)))) < r_min_init:
            continue
        U = total_energy(join_state(pos, np.zeros((3, 2))))  # potential only
        K_needed = E0 - U
        if K_needed <= 0:  # cannot reach E0 (too compact)
            continue
        vel = rng.normal(size=(3, 2))
        vel -= vel.mean(0)  # zero total momentum
        K = 0.5 * (MASSES * (vel**2).sum(-1)).sum()
        vel *= np.sqrt(K_needed / K)
        s0 = join_state(pos, vel)
        assert abs(total_energy(s0) - E0) < 1e-9
        return s0
    raise RuntimeError("no valid initial condition found")


# ------------------------------------------------------------- trajectories
def simulate(s0, T, dt, rtol=1e-10, atol=1e-12, r_escape=6.0, r_close=0.05):
    """Integrate from s0 over [0,T], sampled every dt.
    Returns (traj (n,12), t (n,)) truncated at the first escape / close encounter."""
    t_eval = np.arange(0.0, T + 1e-12, dt)
    sol = solve_ivp(
        rhs, (0.0, T), s0, method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol
    )
    traj = sol.y.T
    bad = (max_radius(traj) > r_escape) | (min_pair_distance(traj) < r_close)
    if bad.any():
        k = int(np.argmax(bad))
        traj = traj[:k]
    return traj, t_eval[: len(traj)]


def generate_dataset(
    n_traj=200,
    T=20.0,
    dt=0.01,
    E0=-0.5,
    seed=0,
    min_len_frac=0.5,
    max_energy_drift=1e-3,
    **kw,
):
    """Generate n_traj trajectories of (up to) T/dt+1 steps. Trajectories that are
    cut too early, or whose total energy drifts by more than max_energy_drift
    from E0 (solver not accurate enough), are discarded and replaced.
    Returns list of arrays (len_i, 12)."""
    rng = np.random.default_rng(seed)
    min_len = int(min_len_frac * T / dt)
    out, tried = [], 0
    while len(out) < n_traj:
        tried += 1
        s0 = sample_initial_condition(rng, E0=E0)
        traj, _ = simulate(s0, T, dt, **kw)
        if (
            len(traj) >= min_len
            and np.abs(total_energy(traj) - E0).max() <= max_energy_drift
        ):
            out.append(traj)
    print(f"generated {n_traj} trajectories ({tried} attempted)")
    return out


def pad_stack(trajs):
    """List of (len_i,12) -> array (n, Lmax, 12) padded with NaN, plus lengths."""
    L = max(len(t) for t in trajs)
    arr = np.full((len(trajs), L, 12), np.nan)
    for i, t in enumerate(trajs):
        arr[i, : len(t)] = t
    return arr, np.array([len(t) for t in trajs])
