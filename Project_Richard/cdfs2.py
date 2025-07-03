import numpy as np
from scipy.special import j1

def build_cdfs(S, Z, INV_R_GRID=500, INV_T_GRID=2000):
    # 1) dimensionless grids on [0,1]
    r_star_grid = np.linspace(0, 1, INV_R_GRID)
    t_star_grid = np.linspace(0, 1, INV_T_GRID)

    # 2) shared exponential factor e^{-z^2 S/2}
    exp_term = np.exp(- Z**2 * (S/2))

    # 3) modal prefactors
    A_den = 2.0 / (Z * j1(Z))         
    A_num = 2.0 / (Z * (j1(Z)**2))    

    # --- build spatial‐exit numerator on r* grid ---
    # outer‐product: modes × grid
    #   A_num[n]*exp_term[n] * J1(z_n * r*) * r*
    num_terms = (A_num[:,None] * exp_term[:,None]) * j1(Z[:,None] * r_star_grid[None,:]) * r_star_grid[None,:]
    cdf_spatial_num  = num_terms.sum(axis=0)  

    # 4) survival probability p_surv = sum_n A_den[n] * exp_term[n]
    p_surv0 = (A_den * exp_term).sum()

    # 5) normalized spatial CDF F_spatial*(r*) on [0,1]
    cdf_spatial_star = cdf_spatial_num / p_surv0

    # --- build raw exit‐time CDF on t* grid (unnormalized) ---
    temp_terms     = A_den[:,None] * np.exp(-Z[:,None]**2 * (S * t_star_grid[None,:]/2))
    raw_cdf_temp   = 1.0 - temp_terms.sum(axis=0)

    return (r_star_grid, cdf_spatial_star, p_surv0), (t_star_grid, raw_cdf_temp)