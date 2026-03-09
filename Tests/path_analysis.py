import math
import time
import random
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from scipy.special import jv, jvp, gamma, loggamma
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
font = font_manager.FontProperties(style='normal', size=20)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

from bessel_zeros import get_bessel_zeros
from CDFs import build_cdfs as build_cdfs

# ── PARAMETERS ──────────────────────────────────────────────────────────────
T_total = 10.0 # Terminal horizon
dims     = [2, 10, 50, 100] # dimensions
def s_range(d):
    if d == 100:
        S = np.arange(0.0073, 0.0123, 0.001) # cylinder scaling parameter
    elif d == 50:
        S = np.arange(0.0129, 0.0264, 0.002) # cylinder scaling parameter
    elif d == 10:
        S = np.arange(0.04, 0.17, 0.01) # cylinder scaling parameter
    elif d == 2:
        S = np.arange(0.1, 1.1, 0.1) # cylinder scaling parameter
    return S
N_ZEROS = 200 # number of terms in the Fourier-Bessel series
INV_R   = 2000 # table for inverse distribution function for distance
INV_T   = 2000 # table for inverse distribution function for time
K       = 1.1
N_PATHS = 1_000 # number of sample paths to simulate
tol     = 0 # stopping criteria
_eps    = np.finfo(np.float64).eps # small number to avoid division by zero
# nu  = DIM/2.0 - 1.0 # order of Bessel function
epsilon = 1e-4 # small threshold to stop WoHB

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

def simulate_path_WoHB(DIM: int, T_rem: float,center) -> np.ndarray:
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


def simulate_path_WoC(DIM: int, T_rem: float,center:float,s:float) -> np.ndarray:
    t_0 = 0
    path = np.insert(center, 0, t_0)[None,:]
    while True:
        if T_rem <= tol:
            return path
        

        # R so that t* = T_rem/R^2 = S
        R = math.sqrt(max(T_rem, 0.0) / s)

        # Random direction on unit sphere
        direction = np.random.normal(size=DIM)
        nrm = np.linalg.norm(direction)
        if nrm <= _eps:
            direction = np.zeros(DIM); direction[0] = 1.0 # degenerate case
        else:
            direction /= nrm # a point on unit sphere

        # Degenerate guards first
        if p_surv0 <= 0.0:
            u_e = np.random.rand()
            idx = min(int(u_e * (INV_T - 1)), INV_T - 1)
            t_s = t_star_inv[idx]
            tau = t_s * T_rem
            center += R * direction
            T_rem  -= tau
            t_0 += tau
            path = np.concatenate((path,np.insert(center,0,t_0)[None,:]), axis = 0)
            i += tau
            continue

        if p_surv0 >= 1.0 - 1e-15:
            u_c = np.random.rand()
            idx = min(int(u_c * (INV_R - 1)), INV_R - 1)
            r_s = r_star_inv[idx]
            center += (r_s * R) * direction
            path = np.concatenate((path,np.insert(center,0,t_0+T_rem)[None,:]), axis = 0)
            return path

        # Standard branch
        u = np.random.rand()
        if u < p_surv0:
            u_c = u / p_surv0
            idx = min(int(u_c * (INV_R - 1)), INV_R - 1)
            r_s = r_star_inv[idx]
            center += (r_s * R) * direction
            path = np.concatenate((path,np.insert(center,0,t_0+T_rem)[None,:]), axis = 0)
            return path
        else:
            u_e = (u - p_surv0) / p_exit0
            idx = min(int(u_e * (INV_T - 1)), INV_T - 1)
            t_s = t_star_inv[idx]
            tau = t_s * T_rem
            center += R * direction
            T_rem  -= tau
            t_0 += tau
            path = np.concatenate((path,np.insert(center,0,t_0)[None,:]), axis = 0)

dict_WoC = {d: {} for d in dims}
dict_WoHB = {d: {} for d in dims}
for d in dims:
    S = s_range(d)
    for s in S:
        print(f"Running for s = {s} and dimension d = {d}…")
        # 1) Bessel zeros
        # print("Computing Bessel function zeros…")
        zeros = get_bessel_zeros(d, N_ZEROS)
        # print(f" Retrieved {len(zeros)} zeros for ν={d/2 -1}")

        # 2) Build CDFs (validated)
        r_star, cdf_r, p_surv0, t_star, raw_t = build_cdfs(d, s, zeros, INV_R, INV_T)
        p_surv0 = float(np.clip(p_surv0, 0.0, 1.0))
        p_exit0 = 1.0 - p_surv0
        # print(f"  Survival Probability p_surv0 = {p_surv0:.6e}, Exit Probability p_exit0 = {p_exit0:.6e}")

        u_r = np.linspace(0.0, 1.0, INV_R)
        cdf_r = np.maximum.accumulate(np.clip(cdf_r, 0.0, 1.0))
        r_star_inv = np.interp(u_r, cdf_r, r_star)


        u_t = np.linspace(0.0, 1.0, INV_T)
        if p_exit0 > 0.0:
            cond_exit_cdf = np.maximum.accumulate(np.clip(raw_t / p_exit0, 0.0, 1.0))
            t_star_inv = np.interp(u_t, cond_exit_cdf, t_star)
        else:
            t_star_inv = np.zeros_like(u_t)
            
            
        sample_paths_WoC = []
        for i in range(N_PATHS):
            x = np.full(d, 0., dtype=np.float64)
            woc = simulate_path_WoC(d,T_total,x,s)
            sample_paths_WoC.append(woc)
        times_WoC = []
        for sp in sample_paths_WoC:
            for st in sp:
                if st[0]>0 and st[0] < T_total:
                    times_WoC.append(st[0])
        dict_WoC[d][s] = times_WoC
    sample_paths_WoHB = []
    for i in range(N_PATHS):
        x = np.full(d, 0., dtype=np.float64)# the starting point
        wohb = simulate_path_WoHB(d,T_total,x)
        sample_paths_WoHB.append(wohb) 
    times_WoHB = []
    for sp in sample_paths_WoHB:
        for s in sp:
            if s[0]>0 and s[0] < T_total:
                times_WoHB.append(s[0])
    dict_WoHB[d]['wohb'] = times_WoHB


with open(f"path_times_WoC.json", "w") as json_file:
    json.dump(dict_WoC, json_file, indent=4)

with open(f"path_times_WoHB.json", "w") as json_file:
    json.dump(times_WoHB, json_file, indent=4)



# length_WoC = []
# for s in sample_paths_WoC:
#     length_WoC.append(s.shape[0])
# length_WoC = np.array(length_WoC)

# dict_WoC[S] = length_WoC.tolist()


# length_WoHB = []
# for s in sample_paths_WoHB:
#     length_WoHB.append(s.shape[0])
# length_WoHB = np.array(length_WoHB)
# dict_WoHB[S] = length_WoHB.tolist()



    
# with open(f"path_length_WoC_{DIM}.json", "w") as json_file:
#     json.dump(dict_WoC, json_file, indent=4)
# with open(f"path_length_WoHB_{DIM}.json", "w") as json_file:
#     json.dump(dict_WoHB, json_file, indent=4)
    

