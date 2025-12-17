import math
import time
import random
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from bessel_zeros import get_bessel_zeros
from scipy.special import jv, jvp, gamma, loggamma
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
font = font_manager.FontProperties(style='normal', size=20)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

from bessel_zeros import get_bessel_zeros
from CDFs3_working import build_cdfs as build_cdfs
import seaborn as sns

# ── PARAMETERS ──────────────────────────────────────────────────────────────
T_total = 10.0
DIM     = 15
S       = [0.005*i for i in range(1,5)]  # cylinder nondimensional step
N_ZEROS = 200
INV_R   = 2000
INV_T   = 2000
K       = 1.1
s       = 1.0
N_PATHS = 50_000
tol     = 1e-8
_eps    = np.finfo(np.float64).eps
nu  = DIM/2.0 - 1.0

def simulate_whole_path(T_rem: float,s_:float) -> np.ndarray:
    center = np.full(DIM, s_, dtype=np.float64)
    t_0 = 0
    path = np.insert(center, 0, t_0)[None,:]
    i = 0
    while True:
        if T_rem <= tol:
            return path

        # R so that t* = T_rem/R^2 = S
        R = math.sqrt(max(T_rem, 0.0) / s_)

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

path_dict = {}
for s_ in S:
    print(f"Running for S = {s_} ...")
    # 1) Bessel zeros
    print("Computing Bessel function zeros…")
    zeros = get_bessel_zeros(DIM, N_ZEROS)
    print(f" Retrieved {len(zeros)} zeros for ν={DIM/2 -1}")

    # 2) Build CDFs (validated)
    r_star, cdf_r, p_surv0, t_star, raw_t = build_cdfs(DIM, s_, zeros, INV_R, INV_T)
    p_surv0 = float(np.clip(p_surv0, 0.0, 1.0))
    p_exit0 = 1.0 - p_surv0
    print(f"  Survival Probability p_surv0 = {p_surv0:.6e}, Exit Probability p_exit0 = {p_exit0:.6e}")
    
    u_r = np.linspace(0.0, 1.0, INV_R)
    cdf_r = np.maximum.accumulate(np.clip(cdf_r, 0.0, 1.0))
    r_star_inv = np.interp(u_r, cdf_r, r_star)
    
    
    u_t = np.linspace(0.0, 1.0, INV_T)
    if p_exit0 > 0.0:
        cond_exit_cdf = np.maximum.accumulate(np.clip(raw_t / p_exit0, 0.0, 1.0))
        t_star_inv = np.interp(u_t, cond_exit_cdf, t_star)
    else:
        t_star_inv = np.zeros_like(u_t)
        
        
    sample_paths = []
    for i in range(N_PATHS):
        # if i and (i % 25_000 == 0):
        #     print(f"Simulated {i} paths…")
        sample_paths.append(simulate_whole_path(T_total,s_))
                
    length = []
    for s in sample_paths:
        length.append(s.shape[0])
    length = np.array(length)
    path_dict[s_] = length.tolist()
    
with open(f"path_length_{DIM}.json", "w") as json_file:
    json.dump(path_dict, json_file, indent=4) 
    # # Plotting the histogram
    # _, counts = np.unique(length, return_counts=True)
    # # Find the maximum frequency
    # highest_density = counts.max()/length.size
    # f = plt.figure(figsize=(8, 5),dpi=200)
    # sns.histplot(x=length, kde=True, bins=50,label='Path Lengths Histogram',stat= 'density')
    # # plt.yscale('log')
    # plt.vlines(np.mean(length),0,highest_density, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {np.mean(length):.2f}')
    # plt.xlabel('Number of Steps in Path', fontproperties=font)
    # plt.ylabel('Frequency (log scale)', fontproperties=font)
    # plt.title(f'Histogram of Path Lengths for $d={DIM}$ and $S={s_:.2f}$', fontproperties=font)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(prop=font)
    # # plt.show()
    # f.savefig(f'{DIM}_{s_:.2f}.pdf',format="pdf",dpi=600);