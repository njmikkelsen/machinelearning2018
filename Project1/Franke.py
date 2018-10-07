import sys
import numpy as np
from LinearRegression import *
from misclib import *

# Franke parameters
N      = 600   # number of data points
N_surf = 200   # number of surface grid points
x0     = 0     # minimum x value
x1     = 1     # maximum x value
y0     = 0     # minimum y value
y1     = 1     # maximum y value
sigma  = 0.0   # standard deviation of noise
deg    = 5     # polynomial degree
name   = "poly_deg{:d}".format(deg)


A = Franke_PolynomialApproximation(N,sigma,x0,x1,y0,y1)
A.add_polynomial(5)
A(technique="Kfold",K=10)
A.plot_model()



