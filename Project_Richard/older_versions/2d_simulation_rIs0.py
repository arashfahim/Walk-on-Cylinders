import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1
from bessel_zeros import get_bessel_zeros

R = 1
t = 1 
alpha = 0.5
N = 50
r = 0  

zn = get_bessel_zeros(2, N)

rho_vals = np.linspace(0.01, R, 200)
U_r0 = np.zeros(len(rho_vals))

for i, rho in enumerate(rho_vals):
    u_sum = 0
    for n in range(N):
        lambda_n = zn[n] / R
        A_n = (2 * rho * j1(lambda_n * rho)) / (R * zn[n] * (j1(zn[n])**2))
        u_sum += A_n * np.exp(-alpha * lambda_n**2 * t) * j0(0)  
    U_r0[i] = u_sum

plt.figure(figsize=(8, 5))
plt.plot(rho_vals, U_r0, lw=2)
plt.xlabel('Initial Heat Radius $\\rho$')
plt.ylabel('Temperature at $r = 0$, $u(0, t=1)$')
plt.title('Temperature at Center $u(0, 1)$ vs. Initial Radius $\\rho$')
plt.grid(True)
plt.tight_layout()
plt.show()