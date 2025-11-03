import numpy as np
from scipy.special import jv

def build_cdfs(dim, S, zeros, INV_R=2000, INV_T=2000):
    nu   = dim/2 - 1
    x   = np.asarray(zeros)
    lam = x**2 / 2            # eigenvalues of (1/2)Δ on unit ball

    # 1)  grids on time and radius
    r_star = np.linspace(0.0, 1.0, INV_R)
    t_star = np.linspace(0.0, 1.0, INV_T)

    # 2) modal prefactors
    A_den = 2.0 / (x * jv(nu+1, x))         # time‐series coefficient
    A_num = 2.0 / (x * (jv(nu+1, x)**2))    # space‐series coefficient

    # 3) exponentials at full step S
    expS = np.exp(-lam * S)                # e^{-λ_n S}

    # 4) build unnormalized *spatial* numerator on r* grid
    #    N(r*) = Σ_n [ A_num[n]·expS[n]·J_{nu+1}(x_n·r*)·(r*)^{nu+1} ]
    Jp1    = jv(nu+1, np.outer(x, r_star))   # shape (N_roots, INV_R)
    rwt    = r_star[None,:]**(nu+1)          # (r*)^{nu+1}
    num    = (A_num[:,None] * expS[:,None]) * Jp1 * rwt
    cdf_num = num.sum(axis=0)

    # 5) survival probability at t*=S
    p_surv0 = (A_den * expS).sum()

    # 6) normalized spatial CDF on [0,1]
    cdf_r = cdf_num / p_surv0

    # 7) raw (unnormalized) *time*‐CDF on t* grid
    #    F_raw(t*) = 1 - Σ_n [ A_den[n]·exp(-λ_n·(S·t*)) ]
    exp_tS = np.exp(-np.outer(lam * S, t_star))  # shape (N_roots, INV_T)
    raw_t  = 1.0 - (A_den[:,None] * exp_tS).sum(axis=0)

    return r_star, cdf_r, p_surv0, t_star, raw_t