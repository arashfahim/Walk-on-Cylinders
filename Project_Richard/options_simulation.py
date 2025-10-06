import numpy as np
from scipy.stats import norm

def bachelier_formula(DIM, T, s, K):
# same parameters as your sim   
    sigma   = np.sqrt(T)

    d = (K - s) / sigma        

    term1 = sigma * norm.pdf(d)
    term2 = (s - K) * (1 - norm.cdf(d))

    C_integral = term1 + term2
    
    return C_integral