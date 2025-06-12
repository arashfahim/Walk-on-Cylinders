import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

# Creates an array of 1000 evenly spaced values from 0 to 20.
x = np.linspace(0, 20, 1000)

# Plot Bessel functions for ν = 0, 1, 2, 3
orders = [0, 1, 2, 3]
for n in orders:
    plt.plot(x, jv(n, x), label=f'$J_{n}(x)$')

plt.title("Bessel Functions of the First Kind")
plt.xlabel("x")
#shows the Bessel function symbolically using LaTeX syntax.
#\nu for the greek letter v
plt.ylabel("$J_\\nu(x)$")
plt.axhline(0, color='gray', linestyle='--')
#Activates the legend and puts all the label= values into a visible legend box on the plot.
plt.legend(title="Order of Bessel Fuctions")
plt.grid(True)
plt.savefig("bessel_plot.pdf", bbox_inches='tight')

