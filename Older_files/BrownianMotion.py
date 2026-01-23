import numpy as np
from scipy.special import j0, j1, jn_zeros
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def survival_function(t_vals, Z, R):
    """
    Survival function P(τ_R > t) for Brownian motion in a disk radius R.
    """
    S = []
    for t in t_vals:
        s = 0.0
        for n in range(len(Z)):
            coeff = 2 / (Z[n] * j1(Z[n]))
            s += coeff * np.exp(- (Z[n]**2) * t / (2 * R**2))
        S.append(s)
    return np.array(S)

def hitting_cdf(t_vals, Z, R):
    """
    CDF F(t) = P(τ_R ≤ t) = 1 - Survival function.
    """
    return 1 - survival_function(t_vals, Z, R)

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

    survival_prob = cdf_vals[-1]  # P(tau_R ≥ T) is the value at the maximum p_grid

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

    p_inverse = interp1d(cdf_vals, p_grid, kind='linear', bounds_error=False, fill_value=(0, R))
    return p_inverse, cdf_vals, p_grid, survival_prob

def generate_sample_from_cdf(p_inverse, num_samples=1000):
    """
    Sample from the distribution using inverse transform sampling
    """
    u = np.random.uniform(0, 1, num_samples)
    return p_inverse(u)

def emprical_cdf(samples, p_grid):
    """
    Compute the empirical CDF from samples
    """
    cdf_empirical = np.zeros_like(p_grid)
    for i, p in enumerate(p_grid):
        cdf_empirical[i] = np.mean(samples <= p)
    return cdf_empirical

# Simulation and plots
T = 1
R = 1
N_terms = 100
num_samples = 1000
Z = jn_zeros(0, N_terms)  # zeros of Bessel function J0

# Time grid
t_vals = np.linspace(0.01, 1, 300)

# Calculate survival and hitting CDF
S_vals = survival_function(t_vals, Z, R)
F_vals = hitting_cdf(t_vals, Z, R)
Z = jn_zeros(0, 100)

# Step 1: Compute inverse CDF and theoretical CDF
p_inverse, cdf_theoretical, p_grid, survival_prob = compute_inverse_cdf(T, R)

# Step 2: Generate samples
samples = generate_sample_from_cdf(p_inverse, num_samples)

# Step 3: Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
cdf_empirical = emprical_cdf(samples, p_grid)
plt.plot(p_grid, cdf_theoretical, label='Theoretical CDF', color='blue')
plt.plot(p_grid, cdf_empirical, label='Empirical CDF', color='orange', linestyle='--')
plt.xlabel('Distance |B_T|')
plt.ylabel('CDF F(p)')
plt.title('Conditional CDF of |B_T| (Given Survival)')
plt.grid(True)
plt.legend()

# Plotting
plt.subplot(1,2,2)
plt.plot(t_vals, S_vals, label="Survival Function P(τ_R > t)")
plt.plot(t_vals, F_vals, label="Hitting Time CDF P(τ_R ≤ t)")
plt.axvline(x=T, color='black', linestyle='--', label=f'T = {T}')
plt.xlabel("Time t")
plt.ylabel("Probability")
plt.title("Survival Function and Hitting Time CDF for Brownian Motion in Disk")
plt.legend()
plt.grid(True)
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

"""
Generated 1000 samples with:
Mean: 0.5154
Std Dev: 0.2152
Mean ~0.5 → on average, surviving particles end up about 50% of the way from the center to the boundary.
Std dev ~0.2 → most particles are within [0.4, 0.8] radius from the center.
Survival Probability P(τ_R ≥ T) = 0.088890   => There are 8.89% chance that the Brownian motion does not hit the circle of radius R by time T.
Hitting Probability P(τ_R ≤ T) = 0.911110    => There are 91.11% chance that the Brownian motion hits the circle of radius R by time T.
CDF values F(ρ):
F(0.00) = 0.000000       => The CDF at ρ = 0 is always 0.
F(0.25) = 0.138317       => At ρ = 0.25, the CDF is approximately 0.138, meaning about 13.8% of particles are within this distance.
F(0.50) = 0.480508       => At ρ = 0.50, the CDF is approximately 0.481, meaning about 48.1% of particles are within this distance.
F(0.75) = 0.840190       => At ρ = 0.75, the CDF is approximately 0.840, meaning about 84.0% of particles are within this distance.
F(1.00) = 1.000000       => At ρ = 1.0, the CDF is always 1, meaning all particles are within the circle of radius R.






#############################################################################################################################################
# Hitting Time CDF and Survival Function for Brownian Motion
# Hitting Time CDF and Survival Function for Brownian Motion
# This code computes the CDF of the hitting time τ_R for Brownian motion
# to a disk of radius R, and the survival function P(τ_R > t) using the series expansion

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


# --- Parameters ---
R = 1.0            # Radius of the disk
t_star = 1.0       # Final time
dt = 0.001         # Time step for simulation
M = 100            # Number of Brownian motion paths
N_terms = 100      # Number of terms in Bessel series

# --- Survival function using Bessel series ---
def survival_function(t, R=1.0, N_terms=100):
    """
    # Compute P(τ_R > t): probability Brownian motion stays within disk of radius R until time t.
"""
    Z = jn_zeros(0, N_terms)
    sum = 0.0
    for z in Z:
        term = (2 / (z * (j1(z))**2)) * np.exp(- (z**2) * t / (2 * R**2))
        sum += term
    return sum

# --- Time grid for plotting ---
time_grid = np.linspace(0.01, t_star, 200)  # Avoid t = 0
survival_vals = np.array([survival_function(t, R=R, N_terms=N_terms) for t in time_grid])
survival_vals = np.clip(survival_vals, 0, 1)  # Ensure non-negative values
hitting_cdf_vals = 1 - survival_vals

# --- Simulate 2D Brownian motion to get hitting times ---
n_steps = int(t_star / dt)
hitting_times = []

for _ in range(M):
    x, y = 0.0, 0.0
    for step in range(n_steps):
        t = step * dt
        dx, dy = np.random.normal(0, np.sqrt(dt), 2)
        x += dx
        y += dy
        if x**2 + y**2 >= R**2:
            hitting_times.append(t)
            break
    else:
        hitting_times.append(np.nan)  # Did not hit boundary

# Remove NaNs (particles that didn't hit before t_star)
hitting_times = np.array([t for t in hitting_times if not np.isnan(t)])
hitting_times.sort()
# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(time_grid, hitting_cdf_vals, label='P(τ_R ≤ t)', color='crimson')
plt.plot(time_grid, survival_vals, label='P(τ_R > t)', color='navy', linestyle='--')

# Mark and label simulated hitting times
for i, t_hit in enumerate(hitting_times, 1):
    if t_hit <= t_star:
        plt.axvline(t_hit, color='gray', linestyle=':', alpha=0.5)
        plt.text(t_hit, 0.02, f'$t_{{{i}}}$', rotation=90, fontsize=8,
                 verticalalignment='bottom', horizontalalignment='right')
    

plt.xlabel("Time t")
plt.ylabel("Probability")
plt.title(f"Hitting Time CDF and Survival Function (with {len(hitting_times)} Labeled Hits)")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Print statistics
print(f"Simulated {len(hitting_times)} hitting times out of {M} paths.")
if len(hitting_times) > 0:
    print(f"Mean hitting time: {np.mean(hitting_times):.4f}")
    print(f"Std Dev of hitting times: {np.std(hitting_times):.4f}")
for i, t in enumerate(hitting_times, 1):
    print(f"t_{i} = {t:.5f}")

"""
