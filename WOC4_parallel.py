# ★ Set these BEFORE importing numpy to avoid thread over-subscription
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
import time
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from bessel_zeros import get_bessel_zeros
from CDFs3_working import build_cdfs as _build_cdfs
from bachelier_options import bachelier_formula

# ── PARAMETERS ──────────────────────────────────────────────────────────────
T_total = 1.0
DIM     = 10
S       = 1.0/np.sqrt(DIM)  # cylinder nondimensional step
N_ZEROS = 10
INV_R   = 2000
INV_T   = 2000
K       = 1.1
s       = 1.0 #the components of the starting point considered all the same!
N_PATHS = 100_000
tol     = 1e-8
_eps    = np.finfo(np.float64).eps
print(f"Using DIM={DIM}, S={S}, N_ZEROS={N_ZEROS}, INV_R={INV_R}, INV_T={INV_T}, N_PATHS={N_PATHS}")
# ── Helpers ─────────────────────────────────────────────────────────────────
def build_cdfs_checked(dim, S, zeros, INV_R, INV_T):
    out = _build_cdfs(dim, S, zeros, INV_R, INV_T)
    if out is None or not isinstance(out, tuple) or len(out) != 5:
        raise RuntimeError("CDFs3_working.build_cdfs must return 5 outputs")
    r_star, cdf_r, p_surv0, t_star, raw_t = out
    # quick shape/type sanity
    for name, arr in [("r_star", r_star), ("cdf_r", cdf_r), ("t_star", t_star), ("raw_t", raw_t)]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be numpy array, got {type(arr)}")
    if not np.isfinite(p_surv0):
        raise ValueError(f"p_surv0 is not finite: {p_surv0}")
    return r_star, cdf_r, p_surv0, t_star, raw_t


# 1) Bessel zeros
print("Computing Bessel function zeros…")
zeros = get_bessel_zeros(DIM, N_ZEROS)
print(f" Retrieved {len(zeros)} zeros for ν={DIM/2 -1}")

# 2) Build CDFs (validated)
r_star, cdf_r, p_surv0, t_star, raw_t = build_cdfs_checked(DIM, S, zeros, INV_R, INV_T)
p_surv0 = float(np.clip(p_surv0, 0.0, 1.0))
p_exit0 = 1.0 - p_surv0
print(f"  Survival Probability p_surv0 = {p_surv0:.6e}, Exit Probability p_exit0 = {p_exit0:.6e}")

# 3) Precompute inverse-CDF lookup tables (ensure monotone)
u_r = np.linspace(0.0, 1.0, INV_R)
cdf_r = np.maximum.accumulate(np.clip(cdf_r, 0.0, 1.0))
r_star_inv = np.interp(u_r, cdf_r, r_star)

u_t = np.linspace(0.0, 1.0, INV_T)
if p_exit0 > 0.0:
    cond_exit_cdf = np.maximum.accumulate(np.clip(raw_t / p_exit0, 0.0, 1.0))
    t_star_inv = np.interp(u_t, cond_exit_cdf, t_star)
else:
    t_star_inv = np.zeros_like(u_t)

print
# ── Core simulation (unchanged logic) ───────────────────────────────────────
def simulate_path(T_rem: float) -> np.ndarray:
    center = np.full(DIM, s, dtype=np.float64)
    while True:
        if T_rem <= tol:
            return center

        # R so that t* = T_rem/R^2 = S
        R = math.sqrt(max(T_rem, 0.0) / S)

        # Random direction on unit sphere
        direction = np.random.normal(size=DIM)
        nrm = np.linalg.norm(direction)
        if nrm <= _eps:
            direction = np.zeros(DIM); direction[0] = 1.0
        else:
            direction /= nrm

        # Degenerate guards first
        if p_surv0 <= 0.0:
            u_e = np.random.rand()
            idx = min(int(u_e * (INV_T - 1)), INV_T - 1)
            t_s = t_star_inv[idx]
            tau = t_s * T_rem
            center += R * direction
            T_rem  -= tau
            continue

        if p_surv0 >= 1.0 - 1e-15:
            u_c = np.random.rand()
            idx = min(int(u_c * (INV_R - 1)), INV_R - 1)
            r_s = r_star_inv[idx]
            center += (r_s * R) * direction
            return center

        # Standard branch
        u = np.random.rand()
        if u < p_surv0:
            u_c = u / p_surv0
            idx = min(int(u_c * (INV_R - 1)), INV_R - 1)
            r_s = r_star_inv[idx]
            center += (r_s * R) * direction
            return center
        else:
            u_e = (u - p_surv0) / p_exit0
            idx = min(int(u_e * (INV_T - 1)), INV_T - 1)
            t_s = t_star_inv[idx]
            tau = t_s * T_rem
            center += R * direction
            T_rem  -= tau


def mc_option_price() -> float:
    payoffs = np.empty(N_PATHS, dtype=np.float64)
    for i in range(N_PATHS):
        if i and (i % 50_000 == 0):
            print(f"Simulated {i} paths…")
        final = simulate_path(T_total)
        payoffs[i] = max(final[0] - K, 0.0)
    return float(payoffs.mean())


# ── Parallel runner (fixed worker seeding) ──────────────────────────────────
def _worker_seed(seed_base: int):
    """
    Initialize NumPy RNG in each worker with a 32-bit seed derived from:
    parent-provided base, PID, current time, and extra entropy.
    Ensures 0 <= seed < 2**32 for np.random.seed.
    """
    pid  = os.getpid()
    t    = int(time.time() * 1e6)  # microseconds
    seed = (seed_base ^ pid ^ t ^ random.getrandbits(32)) & 0xFFFFFFFF
    if seed == 0:
        seed = 1
    np.random.seed(seed)


def _simulate_batch(n_batch: int) -> float:
    ssum = 0.0
    for _ in range(n_batch):
        final = simulate_path(T_total)
        ssum += max(final[0] - K, 0.0)
    return ssum


def mc_option_price_parallel(n_paths: int, n_workers: int | None = None, batch: int = 50_000) -> float:
    if n_workers is None:
        # leave one core free by default
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    # Build batches
    n_full, rem = divmod(n_paths, batch)
    batches = [batch] * n_full + ([rem] if rem else [])

    base_seed = random.getrandbits(32)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_seed,
        initargs=(base_seed,)
    ) as ex:
        totals = list(ex.map(_simulate_batch, batches, chunksize=1))

    grand_total = float(np.sum(totals))
    return grand_total / n_paths


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    USE_PARALLEL = True  # toggle to False to use single-core version
    t0 = time.time()
    if USE_PARALLEL:
        C_MC = mc_option_price_parallel(N_PATHS, n_workers=None, batch=50_000)
    else:
        C_MC = mc_option_price()
    print(f"Monte Carlo simulation completed in {time.time() - t0:.2f} seconds.")
    C_Bachelier = bachelier_formula(DIM, T_total, s, K)
    print(f"Monte Carlo price: {C_MC:.6f}")
    print(f"Bachelier formula: {C_Bachelier:.6f}")
    print(f"Relative Error:             {100*(C_MC - C_Bachelier)/C_Bachelier:.6f} %")
