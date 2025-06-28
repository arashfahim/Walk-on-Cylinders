import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1, jn_zeros
from scipy.optimize import brentq

def get_bessel_zeros(v, N, step=np.pi):
    from scipy.special import jv
    def f(x): return jv(v, x)
    zeros = []
    x_min, x_max = 0.1, step
    while len(zeros) < N:
        try:
            zero = brentq(f, x_min, x_max)
            zeros.append(zero)
            x_min = zero + 1e-6
            x_max = x_min + step
        except ValueError:
            x_max += step
    return np.array(zeros)

def temporal_cdf(t_vals, R, Z):
    cdf_vals = []
    for t in t_vals:
        s = 0.0
        for n in range(len(Z)):
            coeff = 2 / ((Z[n]) * (j1(Z[n])))
            s += coeff * np.exp(-Z[n]**2 * t/(2*R**2))
        cdf_vals.append(1 - s)
    return np.array(cdf_vals)

def compute_conditional_spatial_cdf(T, R, p_grid, Z):
    cdf_vals = np.zeros_like(p_grid)
    for i, p in enumerate(p_grid):
        if p == 0:
            cdf_vals[i] = 0.0
        else:
            s = 0.0
            for n in range(len(Z)):
                z = Z[n]
                coef = (2 * p) / (R * z * (j1(z) ** 2))
                term = coef * j1(z * p / R) * np.exp(- (z ** 2) * T / (2 * R ** 2))
                s += term
            cdf_vals[i] = s
    survival_prob = cdf_vals[-1]
    cdf_vals /= survival_prob
    return cdf_vals, survival_prob
