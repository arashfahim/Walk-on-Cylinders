# bachelier.py
import numpy as np
from scipy.stats import norm

# same parameters as your sim
D       = 2
s       = 1     
T       = 1
sigma   = np.sqrt(T/D)
K       = 1

d = (K - s) / sigma        

term1 = sigma * norm.pdf(d)
term2 = (s - K) * (1 - norm.cdf(d))

C_integral = term1 + term2

print(f"Bachelier formula: C = {C_integral:.6f}")