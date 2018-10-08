import sys
import numpy as np
from LinearRegression import *
from misclib import *

# terrain data parameters
filename = "SRTM_data_Oslo.tif"
nx       = 2
ny       = 5

# regression parameters
deg    = 8  # polynomial degrees
N_lam  = 100   # number of penalty parameters
lam0   = 1e-9  # lower penalty
lam1   = 5e-2  # upper penalty
K      = 10   # number of iterations in Bootstrap algorithm
OLS    = True  # whether to include OLS regression
Ridge  = True  # whether to include Ridge regression
Lasso  = True  # whether to include Lasso regression

# standard setup
A = Terrain_PolynomialParametrisation(filename,nx,ny)
#A.plot_terrain(True,True)
A.add_polynomial(deg)
A(alpha=0,method="OLS",technique="Bootstrap",K=10)
A.plot_model(method="OLS",deg=deg)
A.plot_error_penalty_dependence(deg,"Bootstrap",10)

