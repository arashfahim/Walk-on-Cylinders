import numpy as np
import matplotlib.pyplot as plt

# Parameters
R = 1.0            # radius of circle
T = 1.0            # total time
dt = 0.001         # time step
n_steps = int(T / dt)
M = 100              # number of paths

np.random.seed(42)

# Storage for paths and hitting times
paths = np.zeros((M, n_steps+1, 2))
hitting_times = np.full(M, np.nan)

# Initial start point for first path
start_x, start_y = 0.0, 0.0

for i in range(M):
    x, y = start_x, start_y
    hit = False
    
    for step in range(n_steps+1):
        if step == 0:
            paths[i, step] = [x, y]
            continue
        
        # Brownian increment
        dx, dy = np.random.normal(0, np.sqrt(dt), 2)
        x_new = x + dx
        y_new = y + dy
        
        if not hit:
            if x_new**2 + y_new**2 > R**2:
                # Record hitting time
                hitting_times[i] = step * dt
                hit = True
                # Stay at last position inside circle
                paths[i, step] = [x, y]
                # Fill remaining steps with last position
                paths[i, step+1:] = [x, y]
                break
            else:
                x, y = x_new, y_new
                paths[i, step] = [x, y]
        else:
            # After hit, stay fixed
            paths[i, step] = [x, y]
    
    # For next path, start at this path's hitting position
    if not np.isnan(hitting_times[i]):
        hit_step = int(hitting_times[i] / dt)
        start_x, start_y = paths[i, hit_step, 0], paths[i, hit_step, 1]
    else:
        # If no hit, start next path at origin or last position
        start_x, start_y = 0.0, 0.0

# Plot circle
theta = np.linspace(0, 2*np.pi, 200)
circle_x = R * np.cos(theta)
circle_y = R * np.sin(theta)

plt.figure(figsize=(8, 8))
plt.plot(circle_x, circle_y, 'k--', label='Circle boundary')

# Plot each path
for i in range(M):
    plt.plot(paths[i, :, 0], paths[i, :, 1], label=f'Path {i+1}')
    if not np.isnan(hitting_times[i]):
        hit_step = int(hitting_times[i]/dt)
        plt.plot(paths[i, hit_step, 0], paths[i, hit_step, 1], 'ro')
        #plt.text(paths[i, hit_step, 0], paths[i, hit_step, 1], f'$t_{{{i+1}}}={hitting_times[i]:.3f}$', fontsize=8)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Chained Brownian Paths inside Circle')
plt.axis('equal')
plt.grid(True)
#plt.legend()
plt.show()
