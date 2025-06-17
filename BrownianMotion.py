# Brownian Motion in a Disk: CDF and Sampling
# This code computes the cumulative distribution function (CDF) of the radial distance
# of a Brownian motion conditioned to stay within a disk of radius R at time T.
# It uses a series expansion based on Bessel functions and performs inverse transform sampling
# to generate samples from this distribution.

import numpy as np
from scipy.special import j0, j1, jn_zeros
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import random

def compute_cdf(T, R, rho_vals, N_terms=100):
    """
    Compute CDF F(rho) = P(|B_T| <= rho | survive inside disk of radius R)
    using the truncated series expansion with N_terms.
    Parameters:
      T       : float, time horizon
      R       : float, disk radius
      rho_vals: array of floats, points where to compute CDF (0 to R)
      N_terms : int, number of terms in series expansion
    Returns:
      cdf_vals: array of floats, CDF values at rho_vals
    """
    alpha = jn_zeros(0, N_terms)  # First N_terms zeros of J0
    cdf_vals = np.zeros_like(rho_vals)
    r = 0.0  # Use the maximum rho for normalization
    for n in range(N_terms):
        Zn = alpha[n]
        # Precompute constants
        denom = R * Zn * (j1(Zn )**2)
        for i, rho in enumerate(rho_vals):
            if rho == 0:
                term = 0
            else:
                term = (2 * rho / denom) * j1(Zn * rho / R) * j0(Zn * r / R) * np.exp(- (Zn**2) * T / (2 * R**2))
            cdf_vals[i] += term
            
    # Ensure CDF starts at 0 and ends at 1 (numerical)
    if len(cdf_vals) > 0:
        cdf_vals -= np.min(cdf_vals)  # Normalize to start at 0
        cdf_vals /= np.max(cdf_vals)  # Normalize to end at 1
    return cdf_vals

def generate_sample_from_cdf(T, R, N_terms=50, grid_points=1000):
    """
    Generate a single sample from the distribution of |B_T| conditioned to stay inside disk radius R.
    """
    # Create fine grid of rho
    rho_grid = np.linspace(0, R, grid_points)
    cdf_vals = compute_cdf(T, R, rho_grid, N_terms)

    # Create interpolation function of the CDF
    cdf_interp = interp1d(rho_grid, cdf_vals, kind='linear', bounds_error=False, fill_value=(0,1))

    # Inverse transform sampling:
    u = random.uniform(0, 1)

     # Safety check: handle edges
    eps = 1e-10
    if u <= cdf_vals[0] + eps:
        return rho_grid[0]
    elif u >= cdf_vals[-1] - eps:
        return rho_grid[-1] - eps  # stay within interpolation range

    
    # Define function to find root for inversion: cdf(rho) - u = 0
    def root_func(rho):
        return cdf_interp(rho) - u

    # Find root in [0, R] using Brent's method
    # find p such thatf(p)=u
    # f(a) and f(b) must have opposite signs
    sample_rho = brentq(root_func, rho_grid[0], rho_grid[-1])
    # Return the sampled radial distance
    return sample_rho

# Example usage:
if __name__ == "__main__":
    T = 1.0
    R = 1.0
    print("Sampled values of |B_T|:")
    for _ in range(5):
        sample = generate_sample_from_cdf(T, R)
        print(f"{sample:.4f}")

    # Optional: plot the CDF for visualization
    rho_x = np.linspace(0, R, 500)
    cdf_y = compute_cdf(T, R, rho_x)
    plt.plot(rho_x, cdf_y, label='CDF of |B_T| conditioned on survival')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'CDF $F(\rho)$')
    plt.title(r'CDF of radial distance $|B_T|$ in disk of radius $R$')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("Brownian_Motion.png")
