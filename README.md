This is based on a new reseach that generated path of Brownian motion by the exit time from a cylinder. It is useful for Monte Carlo methods for parabolic PDEs. The main codes are implemented in '''bessel_zeros.py''' and '''CDFs3_working.py'''. 
At the header of your code you need the lines:
from bessel_zeros import get_bessel_zeros
from CDFs3_working import build_cdfs as build_cdfs
