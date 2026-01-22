This is based on a new reseach that generated path of Brownian motion by the exit time from a cylinder. It is useful for Monte Carlo methods for parabolic PDEs. The main codes are implemented in `bessel_zeros.py ` and `CDFs3_working.py`. 
At the header of your code you need the lines:
```
from bessel_zeros import get_bessel_zeros
from CDFs3_working import build_cdfs as build_cdfs
```
In `WOC4.py` we simulate paths of Brownian motion and test our simulation in an option pricing problem. The function that simulates the paths is `def simulate_path(T_rem: float) -> np.ndarray:`
We also need to define these parameters. 
```
# ── PARAMETERS ──────────────────────────────────────────────────────────────
T_total = 1.0 #terminal horizon
DIM     = 25 #dimension
S       = 1.0/np.sqrt(DIM)  # scaling parameter
N_ZEROS = 10 # number of terms in the Fourier-Bessel series
INV_R   = 2000 # table for inverse distribution function for distance
INV_T   = 2000 # table for inverse distribution function for time
K       = 1.1 #strike price
s       = 1.0 #the components of the starting point considered all the same!
N_PATHS = 100_000 # number of sample paths to simulate
tol     = 1e-8 # stopping criteria
_eps    = np.finfo(np.float64).eps # small number to avoid division by zero
```
We implemented a closed-form solution to the option pricing in `bachelier_options`. If you need to use the bench mark, add the following line:
```
from bachelier_options import bachelier_formula
```

