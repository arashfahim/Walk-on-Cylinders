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
DIM     = 100 # dimension
S       = [0.005+0.0001*i for i in range(1,400,10)]  # cylinder scaling parameter
N_ZEROS = 200 # number of terms in the Fourier-Bessel series
INV_R   = 2000 # table for inverse distribution function for distance
INV_T   = 2000 # table for inverse distribution function for time
K       = 1.1
x       = np.full(DIM, 0., dtype=np.float64)# the starting point
N_PATHS = 50_000 # number of sample paths to simulate
tol     = 0 # stopping criteria
_eps    = np.finfo(np.float64).eps # small number to avoid division by zero
nu  = DIM/2.0 - 1.0 # order of Bessel function
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

def simulate_path_WoHB(T_rem: float,center) -> np.ndarray:
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


def simulate_path_WoC(T_rem: float,center:float,s:float) -> np.ndarray:
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
            # i += tau
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

path_dict_WoC = {}
path_dict_WoHB = {}
p_exit = {}
# 1) Bessel zeros
print("Computing Bessel function zeros…")
zeros = get_bessel_zeros(DIM, N_ZEROS)
print(f" Retrieved {len(zeros)} zeros for ν={DIM/2 -1}")
for s_ in S:
    print(f"Running for S = {s_} ...")
    # 2) Build CDFs (validated)
    r_star, cdf_r, p_surv0, t_star, raw_t = build_cdfs(DIM, s_, zeros, INV_R, INV_T)
    p_surv0 = float(np.clip(p_surv0, 0.0, 1.0))
    p_exit0 = 1.0 - p_surv0
    print(f"  Survival Probability p_surv0 = {p_surv0:.6e}, Exit Probability p_exit0 = {p_exit0:.6e}")
    if p_surv0 <= 1e-2:
        print(f"  Skipping S={s_:.6f} due to low survival probability.")
        break
    else:
        u_r = np.linspace(0.0, 1.0, INV_R)
        cdf_r = np.maximum.accumulate(np.clip(cdf_r, 0.0, 1.0))
        r_star_inv = np.interp(u_r, cdf_r, r_star)
        
        
        u_t = np.linspace(0.0, 1.0, INV_T)
        if p_exit0 > 0.0:
            cond_exit_cdf = np.maximum.accumulate(np.clip(raw_t / p_exit0, 0.0, 1.0))
            t_star_inv = np.interp(u_t, cond_exit_cdf, t_star)
        else:
            t_star_inv = np.zeros_like(u_t)
    

        print(f"Simulation of paths for $S=${s_:.6f}")
        length_WoC = [] 
        for i in range(N_PATHS):
            # if i and (i % 25_000 == 0):
            #     print(f"Simulated {i} paths…")
            woc = simulate_path_WoC(T_total,x,s_)
            length_WoC.append(woc.shape[0]-1)   # subtract 1 to account for the initial point in WoC paths

        path_dict_WoC[s_] = length_WoC
        p_exit[s_] = p_exit0
    
    
length_WoHB = []    
sample_paths_WoHB = []
for i in range(N_PATHS):
    # if i and (i % 25_000 == 0):
    #     print(f"Simulated {i} paths…")
    wohb = simulate_path_WoHB(T_total,x)
    length_WoHB.append(wohb.shape[0]-1)   # subtract 1 to account for the initial point in WoHB paths
     
path_dict_WoHB['wohb'] = length_WoHB
    
with open(f"path_length_WoC_{DIM}.json", "w") as json_file:
    json.dump(path_dict_WoC, json_file, indent=4)
with open(f"path_length_WoHB_{DIM}.json", "w") as json_file:
    json.dump(path_dict_WoHB, json_file, indent=4)
with open(f"p_exit_{DIM}.json", "w") as json_file:
    json.dump(p_exit, json_file, indent=4) 
