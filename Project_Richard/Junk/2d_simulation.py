import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1
from bessel_zeros import get_bessel_zeros

R = 1
t = 1 
alpha = 0.5
N = 50

zn = get_bessel_zeros(2, N)

r_vals = np.linspace(0, R, 400)
rho_vals = np.linspace(0.01, R, 200)
U = np.zeros((len(rho_vals), len(r_vals)))

for i, rho in enumerate(rho_vals):
    for j, r in enumerate(r_vals):
        u_sum = 0
        for n in range(N):
            lambda_n = zn[n] / R
            A_n = (2 * rho * j1(lambda_n * rho)) / (R * zn[n] * (j1(zn[n])**2))
            u_sum += A_n * np.exp(-alpha * lambda_n**2 * t) * j0(lambda_n * r)
        U[i, j] = u_sum

plt.figure(figsize=(10, 6))
plt.imshow(U, extent=[0, R, rho_vals[0], rho_vals[-1]], origin='lower',
           aspect='auto', cmap='inferno')
plt.colorbar(label='Temperature $u(r, t=1)$')
plt.xlabel('Radius $r$')
plt.ylabel('Initial Heat Radius $\\rho$')
plt.title('Temperature Distribution $u(r, 1)$ vs. $r$ and $\\rho$')
plt.tight_layout()
plt.show()