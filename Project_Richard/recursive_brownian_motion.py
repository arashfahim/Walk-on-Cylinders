import numpy as np
from bessel_zeros import get_bessel_zeros
from cdfs import temporal_cdf, compute_conditional_spatial_cdf
import matplotlib.pyplot as plt

T_total    = 1
S          = 1       # only for R = sqrt(T_rem/S)
DIM        = 2
N_ZEROS    = 200
INV_T_GRID = 2000      # grid for time‐CDF inversion
INV_R_GRID = 500       # grid for spatial‐CDF inversion

zeros = get_bessel_zeros(DIM, N_ZEROS)

def simulate_path(T_total, S, max_segments=1000):
    T_rem  = T_total
    center = np.zeros(2)
    path   = []

    for _ in range(max_segments):
        if T_rem <= 0:
            break

        # 1) choose cylinder radius
        R = np.sqrt(T_rem / S)
        phi = 2*np.pi*np.random.rand()

        # 2) build physical‐time grid & eval exit‐CDF
        t_grid   = np.linspace(0, T_rem, INV_T_GRID)
        cdf_time = temporal_cdf(t_grid, R, zeros)
        p_exit   = cdf_time[-1]

        # 3) get survival prob from spatial‐CDF
        r_grid, _ = np.linspace(0, R, INV_R_GRID), None
        _, p_surv = compute_conditional_spatial_cdf(T_rem, R, r_grid, zeros)

        # 4) draw and branch
        u = np.random.rand()
        if u < p_surv:
            # ————— SURVIVAL —————
            tau   = T_rem
            # sample interior radius given survival
            u_cond     = u / p_surv
            cdf_r, _   = compute_conditional_spatial_cdf(T_rem, R, r_grid, zeros)
            r_end_phys = np.interp(u_cond, cdf_r, r_grid)
            survived   = True
        else:
            # ————— EXIT —————
            u_exit     = (u - p_surv) / (1 - p_surv)
            # conditional exit‐time CDF = cdf_time / p_exit
            tau   = np.interp(u_exit, cdf_time / p_exit, t_grid)
            r_end_phys = R
            survived   = False

        # 5) record
        start = center.copy()
        end   = start + r_end_phys * np.array([np.cos(phi), np.sin(phi)])
        path.append({
            'start':    start,
            'end':      end,
            'R':        R,
            'u':        u,
            'tau': tau,
            'survived': survived
        })

        # 6) update
        T_rem -= tau
        center = end
        if survived:
            break

    return path

def plot_path(path):
    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(12,5))
    xs,ys=[],[]
    n = len(path)
    for i,seg in enumerate(path):
        sx,sy=seg['start']; ex,ey=seg['end']; R=seg['R']
        xs+=[sx-R,sx+R]; ys+=[sy-R,sy+R]
        edge = 'blue' if i==0 else 'orange' if seg['survived'] else 'black'
        ax0.add_patch(plt.Circle((sx,sy),R,fill=False,edgecolor=edge,lw=1.5))
        if not seg['survived']:
            ax0.plot([sx,ex],[sy,ey],'--',color='red',lw=1.5)
            ax0.plot([ex],[ey],'ro',ms=6)
        else:
            ax0.plot([sx,ex],[sy,ey],'-',color='green',lw=1.5)
            ax0.plot([ex],[ey],marker='o',mfc='none',mec='green',ms=8,mew=1.5)
    ax0.set_aspect('equal')
    ax0.set_xlim(min(xs)-0.1,max(xs)+0.1)
    ax0.set_ylim(min(ys)-0.1,max(ys)+0.1)
    ax0.set_title('Cylinders & Paths')
    ax0.set_xlabel('X (physical)'); ax0.set_ylabel('Y (physical)')

    u_vals = [s['u'] for s in path]
    t_cum  = np.cumsum([s['tau'] for s in path])
    ax1.plot(range(1,n+1), u_vals, 'o-', label='u_i')
    ax1.plot(range(1,n+1), t_cum,  's--', label='cumulative time')
    ax1.set_xticks(range(1,n+1))
    ax1.set_xlabel('Segment Index'); ax1.set_ylabel('Value')
    ax1.set_title('Uniforms & Cumul. Time'); ax1.legend()
    plt.tight_layout(); plt.show()

if __name__=='__main__':
    path = simulate_path(T_total, S)
    for i,seg in enumerate(path,1):
        print(f"Seg {i}: R={seg['R']:.3f}, τ_phys={seg['tau']:.4e}, survived={seg['survived']}")
    plot_path(path)