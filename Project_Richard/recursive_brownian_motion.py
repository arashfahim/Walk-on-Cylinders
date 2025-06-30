import numpy as np
from bessel_zeros import get_bessel_zeros
from cdfs import temporal_cdf, compute_conditional_spatial_cdf
import matplotlib.pyplot as plt

T_total    = 1  # Terminal time     
S          = 1   #T/R^2
DIM        = 2  #dimension
N_ZEROS    = 200 #number of terms in the Bessel series
INV_T_GRID = 2000    # number of points in the time grid for inverse transform sampling
INV_R_GRID = 500      # number of points in the radius (rho) grid for conditional CDF 

# evaluation of the Bessel zeros
zeros = get_bessel_zeros(DIM, N_ZEROS) 


# simulate paths of Brownian motion in 2D
def simulate_path(T_total, S, max_segments=1000):
    T_rem  = T_total # remaining time
    center = np.zeros(2) # start at the origin
    path   = [] 

    for _ in range(max_segments):
        if T_rem <= 0:
            break

        # choose cylinder radius via S
        R   = np.sqrt(T_rem / S)
        phi = 2*np.pi*np.random.rand()

        # build a physical‐time grid [0, T_rem]
        t_grid   = np.linspace(0, T_rem, INV_T_GRID)
        cdf_time = temporal_cdf(t_grid, zeros)
        p_hit    = cdf_time[-1]

        u = np.random.rand()
        if u < p_hit:
            # hit before T_rem: invert on physical t_grid
            tau_phys   = np.interp(u, cdf_time, t_grid)
            survived   = False
            r_end_phys = R
        else:
            # survive full T_rem
            tau_phys   = T_rem
            survived   = True
            u_surv     = (u - p_hit) / (1 - p_hit)
            r_grid     = np.linspace(0, R, INV_R_GRID)
            cdf_r, _   = compute_conditional_spatial_cdf(T_rem, R, r_grid, zeros)
            r_end_phys = np.interp(u_surv, cdf_r, r_grid)

        start = center.copy()
        end   = start + r_end_phys * np.array([np.cos(phi), np.sin(phi)])

        path.append({
            'start':    start,
            'end':      end,
            'R':        R,
            'u':        u,
            'tau_phys': tau_phys,
            'survived': survived
        })

        T_rem -= tau_phys
        if survived:
            break

        center = end

    return path

def plot_path(path):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    xs, ys = [], []
    n = len(path)

    for i, seg in enumerate(path):
        sx, sy = seg['start']
        ex, ey = seg['end']
        R       = seg['R']
        xs += [sx-R, sx+R]
        ys += [sy-R, sy+R]

        edge = 'blue' if i == 0 else 'orange' if seg['survived'] else 'black'
        ax0.add_patch(plt.Circle((sx, sy), R, fill=False, edgecolor=edge, lw=1.5))

        if not seg['survived']:
            ax0.plot([sx, ex], [sy, ey], '--', color='red', lw=1.5)
            ax0.plot([ex], [ey], 'ro', ms=6)
        else:
            ax0.plot([sx, ex], [sy, ey], '-', color='green', lw=1.5)
            ax0.plot([ex], [ey], marker='o', mfc='none', mec='green', ms=8, mew=1.5)

    ax0.set_aspect('equal')
    ax0.set_xlim(min(xs) - 0.1, max(xs) + 0.1)
    ax0.set_ylim(min(ys) - 0.1, max(ys) + 0.1)
    ax0.set_title('Cylinders & Paths')
    ax0.set_xlabel('X (physical)')
    ax0.set_ylabel('Y (physical)')

    u_vals   = [seg['u']        for seg in path]
    t_cum    = np.cumsum([seg['tau_phys'] for seg in path])

    ax1.plot(range(1, n+1), u_vals, 'o-', label='u_i')
    ax1.plot(range(1, n+1), t_cum,  's--', label='cumulative time')
    ax1.set_xticks(range(1, n+1))
    ax1.set_xlabel('Segment Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Uniforms & Cumulative Time')
    ax1.legend()

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    path = simulate_path(T_total, S)
    for i, seg in enumerate(path, 3):
        print(f"Seg {i}: R={seg['R']:.3f}, τ_phys={seg['tau_phys']:.4e}, survived={seg['survived']}")
    plot_path(path)