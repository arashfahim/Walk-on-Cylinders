import numpy as np
from scipy.special import j0, j1, jn_zeros
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def compute_cdf(T, R, p_grid, N_terms=100):
    """
    Compute the unnormalized CDF F(p) = u(T,0,R,p)
    and return both the conditional CDF and survival probability P(tau_R ≥ T)
    parameters:
        T : float, time horizon
        R : float, radius of the circle
        p_grid : array-like, grid of distances |B_T|
        N_terms : int, number of terms in the series expansion
    returns:
        cdf_vals : array-like, CDF values at p_grid
    survival_prob : float, P(tau_R ≥ T)
    """
    Z = jn_zeros(0, N_terms)
    cdf_vals = np.zeros_like(p_grid)

    # Compute the CDF using the series expansion
    # F(p) = sum_{n=0}^{N_terms-1} (2p / (R * Z_n * (j1(Z_n)^2))) * j1(Z_n * p / R) * exp(-Z_n^2 * T / (2 * R^2))
    # where Z_n are the zeros of the Bessel function J_0
    
    for i, p in enumerate(p_grid):
        if p == 0:
            cdf_vals[i] = 0.0
        else:
            series_sum = 0.0
            for n in range(N_terms):
                z = Z[n]
                coef = (2 * p) / (R * z * (j1(z) ** 2))
                term = coef * j1(z * p / R) * np.exp(- (z ** 2) * T / (2 * R ** 2))
                series_sum += term
            cdf_vals[i] = series_sum

    survival_prob = cdf_vals[-1]  # value at p = R (last point) = P(tau_R ≥ T)

    # Normalize the CDF for conditional distribution
    cdf_vals /= survival_prob

    return cdf_vals, survival_prob

def compute_inverse_cdf(T, R, N_terms=100, grid_points=1000):
    """
    Compute the inverse CDF F^(-1)(u) for inverse transform sampling
    """
    p_grid = np.linspace(0, R, grid_points)
    cdf_vals, survival_prob = compute_cdf(T, R, p_grid, N_terms)

    # Ensure CDF is increasing
    if np.any(np.diff(cdf_vals) < 0):
        raise ValueError("CDF not strictly increasing. Try increasing N_terms.")

    p_inverse = interp1d(cdf_vals, p_grid, kind='cubic', bounds_error=False, fill_value=(0, R))
    return p_inverse, cdf_vals, p_grid, survival_prob

def generate_sample_from_cdf(p_inverse, num_samples=1000):
    """
    Sample from the distribution using inverse transform sampling
    """
    u = np.random.uniform(0, 1, num_samples)
    return p_inverse(u)


# Brownian Motion Conditional CDF and PDF Sampling
# Parameters
T = 1.0
R = 1.0
num_samples = 1000

# Step 1: Compute inverse CDF and theoretical CDF
p_inverse, cdf_theoretical, p_grid, survival_prob = compute_inverse_cdf(T, R)

# Step 2: Generate samples
samples = generate_sample_from_cdf(p_inverse, num_samples)


# Step 3: Plot results
plt.figure(figsize=(12, 5))

# CDF plot
plt.subplot(1, 2, 1)
plt.plot(p_grid, cdf_theoretical, label='Theoretical CDF', color='blue')
plt.xlabel('Distance |B_T|')
plt.ylabel('CDF F(p)')
plt.title('Conditional CDF of |B_T| (Given Survival)')
plt.legend()

# PDF plot
plt.subplot(1, 2, 2)
pdf_theoretical = np.gradient(cdf_theoretical, p_grid)
plt.hist(samples, bins=50, density=True, alpha=0.5, label='Sample Histogram (PDF)', color='orange')
plt.plot(p_grid, pdf_theoretical, label='Theoretical PDF', color='blue')
plt.xlabel('Distance |B_T|')
plt.ylabel('Density')
plt.title('PDF: Theoretical vs Sampled (Conditional)')
plt.legend()

plt.tight_layout()
plt.show()

# Statistics
print(f"\nGenerated {num_samples} samples with:")
print(f"Mean: {np.mean(samples):.4f}")
print(f"Std Dev: {np.std(samples):.4f}")

# Survival Probability
print(f"\nSurvival Probability P(τ_R ≥ T) = {survival_prob:.6f}")
print(f"Hitting Probability P(τ_R ≤ T) = {1 - survival_prob:.6f}")

# Evaluate CDF at specific rho values
F_interp = interp1d(p_grid, cdf_theoretical, kind='cubic', bounds_error=False, fill_value=(0, 1))
rho_values = [0.0, 0.25, 0.5, 0.75, 1.0]
print("\nCDF values F(ρ):")
for rho in rho_values:
    print(f"F({rho:.2f}) = {F_interp(rho):.6f}")
