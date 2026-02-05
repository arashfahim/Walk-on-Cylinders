import numpy as np
import math


# ── PARAMETERS ──────────────────────────────────────────────────────────────
T_total = 1.0 #terminal horizon
DIM     = 25 #dimension
epsilon = 1e-4 # small threshold to stop
x       = np.full(DIM, 1., dtype=np.float64)# the starting point
_eps    = np.finfo(np.float64).eps # small number to avoid division by zero
N_PATHS = 100_000 # number of sample paths to simulate


def walk_on_heat_balls_step(t_n, x_n, d, epsilon):
    """
    Implements a single step of the 'Walk on Heat Balls' algorithm 
    described on page 4 of the shared document.
    """
    
    # 1. Define alpha(u, v) - Eq (3.1)
    # This represents the size of the heat ball, constrained by distance to boundary
    # Note: This requires a function to calculate distance to your specific domain boundary
    # dist_to_boundary = get_distance_to_boundary(x_n) 
    alpha = t_n #min(t_n, (np.e / (2 * d)) * (dist_to_boundary**2))
    
    # 2. Generate Rn+1
    # coordinates of independent uniform random vectors
    num_u = (d // 2) + 1 # floor(d/2) + 1   
    u_coords = np.random.uniform(0, 1, num_u) # U(0,1) random variables
    pi_u = np.prod(u_coords) # Π U_i
    
    g_n = np.random.standard_normal() # N(0,1) random variable
    
    term1 = pi_u**(2/d)
    exponent = -(1 - (2/d) * (d // 2)) * (g_n**2)
    r_next = term1 * np.exp(exponent)
    
    # 3. Update Time (T_n+1)
    t_next = t_n - alpha * r_next
    
    # 4. Update Position (X_n+1)
    # psi_d(t) = sqrt(t * log(t^(-d/2)))
    def psi_d(t, dimension):
        return np.sqrt(t * np.log(t**(-dimension/2)))
    
    # Random vector on unit sphere
    v_next = np.random.standard_normal(d)
    v_next /= np.linalg.norm(v_next)
    
    x_next = x_n + 2 * np.sqrt(alpha) * psi_d(r_next, d) * v_next
    
    return t_next, x_next

# def get_distance_to_boundary(x):
#     """
#     Placeholder for domain geometry. 
#     Example: Distance to boundary of a unit ball centered at origin.
#     """
#     return max(0, 1.0 - np.linalg.norm(x))

def simulate_path(T_rem: float,center) -> np.ndarray:
    t_0 = 0# initial time
    path = np.insert(center, 0, t_0)[None,:]# initialize path with starting point
    while True:
        if T_rem <= epsilon:
            return path
        else: 
            tau, end = walk_on_heat_balls_step(T_rem, center, DIM, epsilon) # perform a step
            T_rem  -= tau
            t_0 += tau
            center = end
            path = np.concatenate((path,np.insert(center,0,t_0)[None,:]), axis = 0)   



# ── Main ─────────────────
if __name__ == '__main__':
    sample_paths = []
    for i in range(N_PATHS):
        if (i % 10_000 == 0):
            print(f"Simulated {i} paths…")
        sample_paths.append(simulate_path(T_total,x))

# Example usage for a 3D problem
# d = 3
# t_0, x_0 = 1.0, np.array([0.0, 0.0, 0.0])
# epsilon = 1e-4