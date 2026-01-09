import numpy as np
from scipy.special import hyp1f1, gamma, jv, jn_zeros

def solve_woc_renewal(theta, s, d, n_terms=100, M=64):
    """
    Computes F(theta) using the Talbot method for numerical Laplace inversion.
    
    Parameters:
    theta: float, log-time (ln T)
    s: float, fixed ratio T/R^2
    d: int, dimension
    n_terms: int, number of terms in the Bessel series expansion
    M: int, number of points for Talbot quadrature (precision)
    """
    nu = (d - 2) / 2.0
    # Precompute Bessel zeros and J_{nu+1} values
    zeros = jn_zeros(nu, n_terms)
    j_plus_vals = jv(nu + 1, zeros)
    
    # Constant multiplier for g_hat
    const = s / (2**nu * gamma(nu + 1))

    def g_hat(lam):
        """Laplace transform of the renewal kernel g(z)"""
        # Summation term-by-term
        res = 0j
        for n in range(n_terms):
            sigma_sq = (zeros[n]**2 * s) / 2.0
            term = (zeros[n]**(nu + 1)) / j_plus_vals[n]
            # hyp1f1(a, b, z) is the confluent hypergeometric function 1F1
            res += term * hyp1f1(1, lam + 2, -sigma_sq)
        
        return (const / (lam + 1)) * res

    def F_hat(lam):
        """Laplace transform of the expected steps F(theta)"""
        # Handle the singularity at lambda=0 if necessary
        if np.abs(lam) < 1e-12:
            return 1e12 
        g = g_hat(lam)
        return 1.0 / (lam * (1.0 - g))

    # Talbot Method Parameters
    # Fixed Talbot parameters for robust numerical inversion
    def talbot_inversion(t):
        if t == 0: return 1.0 # Base case: F(0) = 1
        
        res = 0j
        for k in range(M):
            theta_k = -np.pi + (k + 0.5) * (2 * np.pi / M)
            # Contour path s(theta)
            s_theta = M / t * (0.5017 * theta_k * (1/np.tan(0.6407 * theta_k)) - 0.6122 + 0.2645j * theta_k)
            # Derivative of s(theta)
            ds_theta = M / t * (0.5017 * (1/np.tan(0.6407 * theta_k) - 0.6407 * theta_k * (1/np.sin(0.6407 * theta_k)**2)) + 0.2645j)
            
            res += np.exp(s_theta * t) * F_hat(s_theta) * ds_theta
            
        return (res / (2j * np.pi)).real

    return talbot_inversion(theta)

# Example Usage:
# s_val = 0.5, d = 3, theta = ln(10)
avg_steps = solve_woc_renewal(theta=np.log(10), s=0.5, d=3)
print(f"Expected number of steps F(theta): {avg_steps:.4f}")