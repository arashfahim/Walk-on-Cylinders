import numpy as np
import matplotlib.pyplot as plt
from bessel_zeros import get_bessel_zeros
from cdfs2 import build_cdfs
from scipy.stats import norm

T_total    = 1
S          = 1
DIM        = 2
N_ZEROS    = 200
INV_T_GRID = 2000
INV_R_GRID = 2000

zeros = get_bessel_zeros(DIM, N_ZEROS)
(sp_r, cdf_r_star, p_surv0), (sp_t, raw_cdf_t_star) = build_cdfs(
    S, zeros, INV_T_GRID, INV_T_GRID
)
p_exit0 = 1.0 - p_surv0

def simulate_path(T_total, S, max_segments=100):
    T_rem  = T_total
    center = np.zeros(2)
    path   = []
    tol = 1e-3

    for _ in range(max_segments):
        if T_rem <= tol:
            path.append({
                'start':    center.copy(),
                'end':      center.copy(),
                'R':        0.0,
                'u':        None,
                'tau':      T_rem,
                'survived': True
            })
            return path

        # 1) cylinder radius & angle
        R   = np.sqrt(T_rem / S)
        phi = 2*np.pi*np.random.rand()

        # 2) spatial CDF by interpolation
        r_grid = np.linspace(0, R, INV_R_GRID)
        r_star = r_grid / R
        cdf_r  = np.interp(r_star, sp_r, cdf_r_star)
        p_surv = p_surv0

        # 3) branch
        u = np.random.rand()
        if u < p_surv:
            # SURVIVE
            tau        = T_rem
            u_cond     = u / p_surv
            r_end_phys = np.interp(u_cond, cdf_r, r_grid)
            survived   = True
        else:
            # EXIT
            u_exit     = (u - p_surv) / p_exit0
            # normalize the *raw* temporal CDF on-the-fly
            cdf_t_cond = raw_cdf_t_star / p_exit0
            t_star     = np.interp(u_exit, cdf_t_cond, sp_t)
            tau        = t_star * T_rem
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

        # 5) update / early return on survival
        T_rem  -= tau
        center  = end
        if survived:
            return path

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
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')

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

if __name__=='__main__':
    # # ---One Simulation---
    # path = simulate_path(T_total, S)
    # for i, seg in enumerate(path, 1):
    #     print(f"Seg {i}: R={seg['R']:.3f}, τ_phys={seg['tau']:.4e}, survived={seg['survived']}")
    # plot_path(path)

    #---Many Simulations--- (Mean, Variance, ...)
    N = 1_000_000
    final_positions = np.zeros((N, 2))
    segment_counts  = np.zeros(N, dtype=int)

    for i in range(N):
        if(i % 1000 == 0):
            print(i)
        path = simulate_path(T_total, S)
        final_positions[i] = path[-1]['end']
        segment_counts[i]  = len(path)

    mean_pos = final_positions.mean(axis=0)        
    var_pos  = final_positions.var(axis=0)           
    cov_pos  = np.cov(final_positions, rowvar=False)
    print(f"Mean final position: x={mean_pos[0]:f}, y={mean_pos[1]:f}")
    print(f"Variance:            var(x)={var_pos[0]:}, var(y)={var_pos[1]:}")
    print("Covariance matrix:\n", cov_pos)

    avg_segments = segment_counts.mean()
    var_segments = segment_counts.var()
    print(f"Average # of segments:    {avg_segments:.2f}")
    print(f"Variance in # segments:   {var_segments:.2f}")

    plt.figure(figsize=(5,5))
    plt.scatter(final_positions[:,0], final_positions[:,1], s=2, alpha=0.1)
    plt.axis('equal')
    plt.title(f"Final positions (N={N})")
    plt.xlabel("x"); plt.ylabel("y")

    x = final_positions[:,0]
    y = final_positions[:,1]

    for coord, name in ((x,'X'), (y,'Y')):
        plt.figure()
        plt.hist(coord, bins=50, density=True, alpha=0.6, label=f"Empirical {name}")
        xs = np.linspace(coord.min(), coord.max(), 200)
        pdf = norm.pdf(xs, loc=0, scale=np.sqrt(T_total))
        plt.plot(xs, pdf, 'r-', lw=2, label=f"N(0,√{T_total}) PDF")
        plt.title(f"{name}_T")
    plt.show()
