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

def temporal_cdf(t_vals, Z):
    cdf_vals = []
    for t in t_vals:
        s = 0.0
        for n in range(len(Z)):
            coeff = 2 / ((Z[n]) * (j1(Z[n])))
            s += coeff * np.exp(-Z[n]**2 * (t/2))
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

T = 1.0
R = 1.0
N_terms = 100
Z = get_bessel_zeros(0, N_terms)

t_vals = np.linspace(0.001, 2, 300)
F_time = temporal_cdf(t_vals, Z)

p_grid = np.linspace(0, R, 300)
F_conditional_rho, P_survival = compute_conditional_spatial_cdf(T, R, p_grid, Z)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(t_vals, F_time, lw=2)
plt.xlabel(r'$t$')
plt.ylabel(r'$\mathbb{P}(\tau_R < t)$')
plt.title('CDF (Exit Time)')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.axhline(P_survival, color='green', lw=2)
plt.text(0.5, P_survival + 0.02, f"P(τ ≥ T) ≈ {P_survival:.4f}", fontsize=12)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks([])
plt.ylabel('Probability')
plt.title('Survival Probability at T')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(p_grid, F_conditional_rho, lw=2)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\mathbb{P}(|B_T| < \rho \mid \tau_R \geq T)$')
plt.title('Spatial CDF')
plt.grid(True)

plt.tight_layout()
plt.show()