import numpy as np
import matplotlib.pyplot as plt

R = 1.0       # radius of circle
T = 1.0        # total time
dt = 0.001
n_steps = int(T / dt)
M = 100       # number of paths

# it makes your code produce the same random numbers every time you run it.
"""
np.random.seed(42)

example how it works
print(np.random.normal(0, 1, 3))
# Output will always be the same every time you run this cell:
# [ 0.49671415 -0.1382643   0.64768854]
"""


"""
M: number of simulated Brownian motion paths

n_steps + 1: number of time steps in each path (including initial time 
t=0)

2: for the 2 coordinates —x and 𝑦
"""
paths = np.zeros((M, n_steps+1, 2))  # (x,y)

"""
The following code is a 1D array of length M that fills with Nan,
nan implies not hit the boundary
"""
hitting_times = np.full(M, np.nan)

for i in range(M):
    x, y = 0.0, 0.0  # start at center
    paths[i, 0] = [x, y]
    hit = False
    for step in range(1, n_steps+1):
        dx, dy = np.random.normal(0, np.sqrt(dt), 2)
        """
        The first argument 0 is the mean of the normal distribution.

        The second argument np.sqrt(0.001) is the standard deviation (square root of variance 0.001).

        The third argument 2 specifies the shape or size of the output array.

        So here, 2 means you want to generate 2 independent random samples from the
        normal distribution with mean 0 and standard deviation sqrt(0.001). 
        The output will be a NumPy array with 2 values, for example:

        array([ 0.018, -0.006])
        """
        x_new = x + dx
        y_new = y + dy

        if not hit:
            if x_new**2 + y_new**2 > R**2:
                hitting_times[i] = step * dt
                hit = True
                # Stay at last position inside circle (or record boundary crossing)
                paths[i, step] = [x, y]
                # Stop path: fill remaining steps with last position
                paths[i, step+1:] = [x, y]
                break
            else:
                x, y = x_new, y_new
                paths[i, step] = [x, y]
        else:
            paths[i, step] = [x, y]  # stay at boundary after hit

# Plot paths and circle
theta = np.linspace(0, 2*np.pi, 200)
circle_x = R * np.cos(theta)
circle_y = R * np.sin(theta)

plt.figure(figsize=(12,12))
plt.plot(circle_x, circle_y, 'k--', label='Circle boundary')

for i in range(M):
    """
    paths[i, : ,0] all x-coordinates for path i
    paths[i, : ,1] all y-coordinates for path i
    """
    plt.plot(paths[i,:,0], paths[i,:,1], label=f'Path {i+1}')
    if not np.isnan(hitting_times[i]):
        hit_step = int(hitting_times[i]/dt)
        plt.plot(paths[i, hit_step, 0], paths[i, hit_step, 1], 'ro')  # hitting point
        plt.text(paths[i, hit_step, 0], paths[i, hit_step, 1], f'$t_{{{i+1}}}={hitting_times[i]:.3f}$', fontsize=8)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Brownian motion inside circle (absorbing boundary)')
# plo.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# Print hitting times
for i, t_hit in enumerate(hitting_times, 1):
    if np.isnan(t_hit):
        print(f"Path {i}: did NOT hit boundary within time T.")
    else:
        print(f"Path {i}: hit boundary at t = {t_hit:.4f}")


print(np.nanmean(hitting_times))
print(np.nanstd(hitting_times))
