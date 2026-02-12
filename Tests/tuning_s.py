import numpy as np
from scipy.special import jv, jvp, gamma, loggamma

# ── PARAMETERS ──────────────────────────────────────────────────────────────
T_total = 1.0 #terminal horizon
DIM     = 2 #dimension
S       = 25e-3  # scaling parameter
N_ZEROS = 40 # number of terms in the Fourier-Bessel series
INV_R   = 2000 # table for inverse distribution function for distance
INV_T   = 2000 # table for inverse distribution function for time
x       = np.full(DIM, 100., dtype=np.float64)# the starting point
K       = x.mean() #strike price
N_PATHS = 50_000 # number of sample paths to simulate
tol     = 0 # stopping criteria
_eps    = np.finfo(np.float64).eps # small number to avoid division by zero
print(f"Using DIM={DIM}, S={S}, N_ZEROS={N_ZEROS}, INV_R={INV_R}, INV_T={INV_T}, N_PATHS={N_PATHS}")
# ── Helpers ─────────────────────────────────────────────────────────────────

def dH_s(dim, S, zeros):
    """
    Same API as your original:
      returns r_star, cdf_r, p_surv0, t_star, raw_t
    with numerically safer internals for large dimensions.
    """
    # ----- setup -----
    nu  = dim/2.0 - 1.0
    x   = np.asarray(zeros, dtype=np.float64)                 # z_{ν,n}
    
    lam = (x*x) / 2.0    # eigenvalues of (1/2)Δ on unit ball = z_{ν,n}^2 / 2


    # ----- stable J_{ν+1}(z_{ν,n}) at roots via derivative identity -----
    # At zeros of J_ν, we have J_{ν+1}(z) = - J'_ν(z) [Watson treatise page 45 (4)]
    Jnu1_roots = -jvp(nu, x, 1)  # derivative w.r.t. argument

    # ----- weights (temporal & spatial) -----
    # coef = 1 / (2^{ν-1} Γ(ν+1))
    coef = 1.0 / (2.0**(nu - 1.0) * gamma(nu + 1.0))

    # A_den = coef * x^{ν-1} / J_{ν+1}(x)
    # Use logs for A_den downstream; still keep a non-log version for shapes/checks if needed
    # log A_den = (nu-1)*log x - (nu-1)*log 2 - log Γ(nu+1) - log J_{ν+1}(x)
    eps = np.finfo(np.float64).tiny
    logA_den = (nu - 1.0) * np.log(np.maximum(x, eps)) \
               - (nu - 1.0) * np.log(2.0) \
               - loggamma(nu + 1.0) \
               - np.log(np.maximum(np.abs(Jnu1_roots), eps))
               - np.log(lam)
    # Keep the correct sign for A_den (J_{ν+1} may change sign):
    sign_A_den = np.sign(Jnu1_roots)
    # log with sign: A_den = (coef / J_{ν+1}) * x^{ν-1}; 'coef' part goes into logA_den above implicitly.
    # Since coef > 0, sign is just sign of 1/J_{ν+1} = sign(J_{ν+1})
    # We'll incorporate sign when summing (below we use pure log-sum-exp on magnitudes, so A_den must be ≥0).
    # For time/survival series the standard closed form yields positive terms, so force nonnegative by magnitude:
    # (If your zeros/ordering create alternating signs numerically, taking abs here is the safe practical choice.)
    logA_den_mag = logA_den  # magnitude in log-domain (we treat terms as positive in the survival/time sums)


    # ----- exponentials for full nondimensional step S -----
    expS = np.exp(-lam * S)  # e^{- (z^2/2) S }

    # ----- survival probability p_surv0 with log-sum-exp -----
    # p_surv0 = Σ_n A_den * e^{-lam*S}
    # => log terms = logA_den_mag - lam*S
    log_terms_surv = logA_den_mag - lam * S
    m = np.max(log_terms_surv)
    p_surv0 = float(np.exp(m) * np.sum(sign_A_den*np.exp(log_terms_surv - m))) #Here is where sign_A_den is multiplied back in.
    # Guard against complete underflow
    if not np.isfinite(p_surv0) or p_surv0 <= 0.0:
        p_surv0 = 0.0

