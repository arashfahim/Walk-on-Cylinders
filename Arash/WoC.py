import numpy as np
import matplotlib.pyplot as plt
from bessel_zeros import get_bessel_zeros
from cdfs import temporal_cdf, compute_conditional_spatial_cdf

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

        # 1) cylinder radius & random angle
        R   = np.sqrt(T_rem / S)
        phi = 2*np.pi*np.random.rand()

        # 2) compute spatial‐CDF & survival prob *once*
        r_grid, _    = np.linspace(0, R, INV_R_GRID), None
        cdf_r, p_surv = compute_conditional_spatial_cdf(T_rem, R, r_grid, zeros)

        # 3) branch
        u = np.random.rand()
        if u < p_surv:
            # ——— SURVIVAL ———
            tau        = T_rem
            u_cond     = u / p_surv
            r_end_phys = np.interp(u_cond, cdf_r, r_grid)
            survived   = True

        else:
            # ——— EXIT ———
            # now we need the temporal‐CDF *only* to sample the exit time
            p_exit    = 1 - p_surv
            t_grid    = np.linspace(0, T_rem, INV_T_GRID)
            cdf_time  = temporal_cdf(t_grid, R, zeros)
            u_exit    = (u - p_surv) / p_exit
            tau       = np.interp(u_exit, cdf_time / p_exit, t_grid)
            r_end_phys = R
            survived   = False

        # 4) record
        start = center.copy()
        end   = start + r_end_phys * np.array([np.cos(phi), np.sin(phi)])
        path.append({
            'start':    start,
            'end':      end,
            'R':        R,
            'u':        u,
            'tau':      tau,
            'survived': survived
        })

        # 5) update
        T_rem  -= tau
        center  = end
        if survived:
            break

    return path

def plot_path(path):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    xs, ys = [], []
    for i, seg in enumerate(path):
        sx, sy = seg['start']
        ex, ey = seg['end']
        R       = seg['R']

        xs += [sx-R, sx+R]
        ys += [sy-R, sy+R]

        edge = 'blue' if i==0 else 'orange' if seg['survived'] else 'black'
        ax0.add_patch(plt.Circle((sx,sy), R, fill=False, edgecolor=edge, lw=1.5))

        if seg['survived']:
            ax0.plot([sx,ex], [sy,ey], '-',   color='green', lw=1.5)
            ax0.plot([ex], [ey], marker='o', mfc='none', mec='green', ms=8, mew=1.5)
        else:
            ax0.plot([sx,ex], [sy,ey], '--',  color='red',   lw=1.5)
            ax0.plot([ex], [ey], 'ro',       ms=6)

    ax0.set_aspect('equal')
    ax0.set_xlim(min(xs)-0.1, max(xs)+0.1)
    ax0.set_ylim(min(ys)-0.1, max(ys)+0.1)
    ax0.set_title('Cylinders & Paths')
    ax0.set_xlabel('X (physical)')
    ax0.set_ylabel('Y (physical)')

    u_vals = [s['u'] for s in path]
    t_cum  = np.cumsum([s['tau'] for s in path])
    idx    = np.arange(1, len(path)+1)
    ax1.plot(idx, u_vals, 'o-',  label='u_i')
    ax1.plot(idx, t_cum,  's--', label='cumul. time')
    ax1.set_xticks(idx)
    ax1.set_xlabel('Segment Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Uniforms & Cumul. Time')
    ax1.legend()

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    path = simulate_path(T_total, S)
    # print path details
    for i, seg in enumerate(path):
        print(f"Seg {i}: R={seg['R']:.3f}, τ_phys={seg['tau']:.4e}, survived={seg['survived']}")
    for i, seg in enumerate(path,1):# Did you set start =1?
        print(f"Seg {i}: R={seg['R']:.3f}, τ_phys={seg['tau']:.4e}, survived={seg['survived']}")
    plot_path(path)