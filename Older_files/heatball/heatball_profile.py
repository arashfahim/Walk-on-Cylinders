import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

def psi_d(t, d=2):
    """
    Computes psi_d(t) = sqrt( t * log(t^(-d/2)) )
    """
    val = np.zeros_like(t)
    # Avoid log(0) or log of negative numbers
    mask = (t > 1e-9) & (t < 1.0 - 1e-9)
    
    # Calculate the term inside the square root
    # log(t^(-d/2)) = (-d/2) * log(t)
    term = t[mask] * (-d/2.0) * np.log(t[mask])
    
    # Ensure non-negative values for sqrt
    term = np.maximum(term, 0)
    
    val[mask] = np.sqrt(term)
    return val

def plot_heat_ball(a=1, d=2):
    """
    Plots the Heat Ball S_a for a given dimension d and time scale a.
    """
    # Create a grid for time (s) and angle (theta)
    s_vals = np.linspace(0, a, 200)
    theta_vals = np.linspace(0, 2*np.pi, 100)
    
    S, Theta = np.meshgrid(s_vals, theta_vals)
    
    # Calculate the normalized time ratio t = s/a
    t_ratio = S / a
    
    # Calculate the radius function R(s)
    # |x - y| = 2 * sqrt(a) * psi_d(s/a)
    Radius = np.sqrt(2*a) * psi_d(t_ratio, d=d)
    
    # Convert polar coordinates to Cartesian for the spatial dimensions (y1, y2)
    Y1 = Radius * np.cos(Theta)
    Y2 = Radius * np.sin(Theta)
    Time = S  # The time axis
    
    # Create the 3D plot
    fig = plt.figure(figsize=(4, 4),dpi=300,constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(Time, Y1, Y2,  color='skyblue',  shade=True, alpha=0.5, edgecolor='none')
    
    # Labels and title
    ax.set_xlabel('$y_1$')
    ax.set_ylabel('$y_2$')
    ax.set_zlabel('$s$ (time)')
    # ax.set_title(f'Heat Ball $\mathcal{{S}}_T$ for $d={d}$')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Set axis limits to center the plot
    max_range = max(np.max(Y1) - np.min(Y1), np.max(Y2) - np.min(Y2), np.max(Time) - np.min(Time)) / 2.0
    mid_y = (np.max(Y1) + np.min(Y1)) * 0.5
    mid_z = (np.max(Y2) + np.min(Y2)) * 0.5
    mid_x = (np.max(Time) + np.min(Time)) * 0.5
    
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_xlim(0, a)
    ax.set_axis_off()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'$x_2$')

    # 3. Manually draw lines for the X, Y, and Z axes through the origin
    # X-axis (black line along x-limits, y=0, z=0)
    ax.plot([xlim[0], xlim[1]+0.1], [0, 0], [0, 0], color='k', linestyle='-', linewidth=1.5)
    # Y-axis (black line along y-limits, x=0, z=0)
    ax.plot([0, 0], [ylim[0], ylim[1]], [0, 0], color='k', linestyle='-', linewidth=1.5)
    # Z-axis (black line along z-limits, x=0, y=0)
    ax.plot([0, 0], [0, 0], [zlim[0], zlim[1]], color='k', linestyle='-', linewidth=1.5)
    ax.text(a, 0, 0, r"$T$", color='black', fontsize=12, ha='center', va='bottom');
    ax.text(a+.15, 0, 0, r"$t$", color='black', fontsize=12, ha='center', va='top');
    ax.view_init(elev=30, azim=-85)
    ax.quiver(
        0, 0, 0,         # Start at x=1, y=1
        a+.1, 0, 0,         # Go 1.1 units in x, 5 units in y, 0 units in z
        length=a+.1,        # Length of the arrow
        normalize=True,    # Normalize the arrow length
        arrow_length_ratio=0.05,  # Ratio of the arrow head to the total length
        fc='black',      # Face color (head)
        ec='black'       # Edge color (line)
    )
    plt.tight_layout(rect=[0, 0, 0.7, 0.5])
    plt.savefig('heatball_plot.pdf', dpi=800, bbox_inches='tight')
    
    fig2 = plt.figure(figsize=(8, 4),dpi=300,constrained_layout=True)
    ax1 = fig2.add_subplot(111)
    R = np.sqrt(2*a) * psi_d(s_vals/a, d=d)
    ax1.plot(s_vals, R, color='blue', label=r'$R(t) = 2\sqrt{T}\psi_d(t/T)$')
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$R(t)$')
    ax1.set_title(f'Radius Function $R(t)$ for $d={d}$')
    # ax1.grid()(True)
    ax1.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig('heatball_radius.pdf', dpi=800, bbox_inches='tight')
    
    # plt.show()
    

# Execute the plotting function
if __name__ == "__main__":
    plot_heat_ball(a=1, d=2)