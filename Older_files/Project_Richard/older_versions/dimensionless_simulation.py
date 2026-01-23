import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from bessel_zeros import get_bessel_zeros

d = int(input("Enter Dimension: "))
v = (d - 2) / 2
t = 1
alpha = 0.5
N = 50
epsilon = 1e-14

zn = get_bessel_zeros(d, N)

r_vals = np.linspace(0, 1, 400)
rho_vals = np.linspace(0.01, 1, 200)
U = np.zeros((len(rho_vals), len(r_vals)))

for i, rho in enumerate(rho_vals):
    for j, r in enumerate(r_vals):
        u_sum = 0
        safe_r = r if r > epsilon else epsilon
        for n in range(N):
            z_n = zn[n]
            A_n = (2 * rho**(v + 1) * jv(v + 1, z_n * rho)) / (z_n * (jv(v + 1, z_n)**2))
            u_sum += A_n * np.exp(-alpha * z_n**2 * t) * jv(v, z_n * r) / (safe_r ** v)
        U[i, j] = u_sum

plt.figure(figsize=(10, 6))
plt.imshow(U, extent=[0, 1, rho_vals[0], rho_vals[-1]], origin='lower',
           aspect='auto', cmap='inferno')
plt.colorbar(label='Temperature $u(r, t)$')
plt.xlabel('Dimensionless Radius $r$')
plt.ylabel('Dimensionless Initial Heat Radius $\\rho$')
plt.title(f'Dimensionless Temperature $u(r, t)$ in {d}D Space (t={t})')
plt.tight_layout()
plt.show()