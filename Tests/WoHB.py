import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from bachelier_options import bachelier_formula
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ── PARAMETERS ──────────────────────────────────────────────────────────────
T_total = 1.0 #terminal horizon
DIM     = 2 #dimension
import time
import random
epsilon = 1e-4 # small threshold to stop
x       = np.full(DIM, 1., dtype=np.float64)# the starting point
_eps    = np.finfo(np.float64).eps # small number to avoid division by zero
N_PATHS = 100_000 # number of sample paths to simulate
K = 1.1 #strike price


def walk_on_heat_balls_step(t_n, x_n, d, epsilon):
    """
    Implements a single step of the 'Walk on Heat Balls' algorithm 
    described on page 4 of the shared document.
    """
    
    # 1. Define alpha(u, v) - Eq (3.1)
    # This represents the size of the heat ball, constrained by distance to boundary
    # Note: This requires a function to calculate distance to your specific domain boundary
    # dist_to_boundary = get_distance_to_boundary(x_n) 
    alpha = t_n #min(t_n, (np.e / (2 * d)) * (dist_to_boundary**2))
    
    # 2. Generate Rn+1
    # coordinates of independent uniform random vectors
    num_u = (d // 2) + 1 # floor(d/2) + 1   
    u_coords = np.random.uniform(0, 1, num_u) # U(0,1) random variables
    pi_u = np.prod(u_coords) # Π U_i
    
    g_n = np.random.standard_normal() # N(0,1) random variable
    
    term1 = pi_u**(2/d)
    exponent = -(1 - (2/d) * (d // 2)) * (g_n**2)
    r_next = term1 * np.exp(exponent)
    
    # 3. Update Time (T_n+1)
    t_next = t_n - alpha * r_next
    
    # 4. Update Position (X_n+1)
    # psi_d(t) = sqrt(t * log(t^(-d/2)))
    def psi_d(t, dimension):
        return np.sqrt(t * np.log(t**(-dimension/2)))
    
    # Random vector on unit sphere
    v_next = np.random.standard_normal(d)
    v_next /= np.linalg.norm(v_next)
    
    x_next = x_n + 2 * np.sqrt(alpha) * psi_d(r_next, d) * v_next
    
    return t_next, x_next

# def get_distance_to_boundary(x):
#     """
#     Placeholder for domain geometry. 
#     Example: Distance to boundary of a unit ball centered at origin.
#     """
#     return max(0, 1.0 - np.linalg.norm(x))

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
        final = simulate_path(T_total, x)
        ssum += max(final[0] - K, 0.0)
    return ssum

def simulate_path(T_rem: float,center) -> np.ndarray:
    t_0 = 0# initial time
    path = np.insert(center, 0, t_0)[None,:]# initialize path with starting point
    while True:
        if T_rem <= epsilon:
            return path
        else: 
            tau, end = walk_on_heat_balls_step(T_rem, center, DIM, epsilon) # perform a step
            T_rem  -= tau
            t_0 += tau
            center = end
            path = np.concatenate((path,np.insert(center,0,t_0)[None,:]), axis = 0)   

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

def mc_option_price() -> float:
    payoffs = np.empty(N_PATHS, dtype=np.float64)
    for i in range(N_PATHS):
        if i and (i % 50_000 == 0):
            print(f"Simulated {i} paths…")
        final = simulate_path(T_total,x)
        payoffs[i] = max(final[0] - K, 0.0)
    return float(payoffs.mean())

# ── Main ─────────────────
if __name__ == '__main__':
    sample_paths = []
    for i in range(N_PATHS):
        if (i % 10_000 == 0):
            print(f"Simulated {i} paths…")
        sample_paths.append(simulate_path(T_total,x))
    
    length = []
    for s in sample_paths:
        length.append(s.shape[0])
    length = np.array(length)

    USE_PARALLEL = False  # toggle to False to use single-core version
    t0 = time.time()
    if USE_PARALLEL:
        C_MC = mc_option_price_parallel(N_PATHS, n_workers=None, batch=50_000)
    else:
        C_MC = mc_option_price()
        
    print(f"Monte Carlo simulation completed in {time.time() - t0:.2f} seconds.")
    C_Bachelier = bachelier_formula(DIM, T_total, x.mean(), K)
    print(f"Monte Carlo price: {C_MC:.6f}")
    print(f"Bachelier formula: {C_Bachelier:.6f}")
    print(f"Relative Error:             {100*(C_MC - C_Bachelier)/C_Bachelier:.6f} %")
    
# Example usage for a 3D problem
# d = 3
# t_0, x_0 = 1.0, np.array([0.0, 0.0, 0.0])
# epsilon = 1e-4