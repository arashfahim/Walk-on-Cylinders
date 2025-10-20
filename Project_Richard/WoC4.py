import numpy as np
import math
from scipy.stats import norm
from bessel_zeros import get_bessel_zeros
from cdfs3 import build_cdfs
from options_simulation import bachelier_formula

# ── PARAMETERS ──────────────────────────────────────────────────────────────
T_total    = 1.0
DIM        = 1
# Choose R so that T_total/R^2 = S = DIM  ⇒ R = √(T_total/DIM)
S          = DIM #Why is S=DIM?

N_ZEROS    = 20     # increase for accuracy
INV_R      = 50
INV_T      = 50
K          = 1.1
s          = 1.0
N_PATHS    = 10_000
tol        = 1e-8

# 1) Precompute zeros of J_{ν}, ν = DIM/2 - 1
zeros = get_bessel_zeros(DIM, N_ZEROS)

# 2) Build the spatial & temporal CDFs once
r_star, cdf_r, p_surv0, t_star, raw_t = build_cdfs(
    DIM, S, zeros, INV_R, INV_T
)
p_exit0 = 1.0 - p_surv0

# 3) Precompute inverse‐CDF lookup tables
u_r        = np.linspace(0.0, 1.0, INV_R)
r_star_inv = np.interp(u_r, cdf_r, r_star)           # u → r*
u_t        = np.linspace(0.0, 1.0, INV_T)
t_star_inv = np.interp(u_t, raw_t/p_exit0, t_star)   # u → t*

def simulate_path(T_rem):
    """
    Simulate one “walk‐on‐cylinders” path in a d-ball until exit or
    survival to time T_rem. Returns the final position vector.
    """
    center = np.full(DIM, s)
    while True:
        # 1) If almost no time remains, return current position
        if T_rem <= tol:
            return center

        # 2) Cylinder radius R so that t* = T_rem / R^2 = S
        R = math.sqrt(T_rem / S)

        # 3) Sample a random direction on the unit sphere in R^d
        direction = np.random.normal(size=DIM)
        direction /= np.linalg.norm(direction)

        # 4) Draw a uniform to decide survive vs exit
        u = np.random.rand()
        if u < p_surv0:
            # → Survive entire remaining time
            u_c = u / p_surv0
            idx = min(int(u_c * (INV_R - 1)), INV_R - 1)
            r_s = r_star_inv[idx]
            center += (r_s * R) * direction
            return center
        else:
            # → Exit before end
            u_e = (u - p_surv0) / p_exit0
            idx = min(int(u_e * (INV_T - 1)), INV_T - 1)
            t_s = t_star_inv[idx]
            # Physical exit time τ = t* · T_rem
            tau = t_s * T_rem
            center += R * direction
            T_rem   = T_rem - tau
            # repeat until survival

def mc_option_price():
    payoffs = np.empty(N_PATHS)
    for i in range(N_PATHS):
        if i and i % 5_000 == 0:
            print(f"Simulated {i} paths…")
        final = simulate_path(T_total)
        payoffs[i] = max(final.mean() - K, 0)
    return payoffs.mean()

if __name__ == '__main__':
    C_MC        = mc_option_price()
    C_Bachelier = bachelier_formula(DIM, T_total, s, K)
    print(f"Monte Carlo price: {C_MC:.6f}")
    print(f"Bachelier formula:  {C_Bachelier:.6f}")
    print(f"Error:              {C_MC - C_Bachelier:.6f}")