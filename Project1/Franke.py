import numpy as np
from LinearRegression import *
from misclib import *

# Franke parameters
N      = 800   # number of data points
N_surf = 500   # number of surface grid points
x0     = 0     # minimum x value
x1     = 1     # maximum x value
y0     = 0     # minimum y value
y1     = 1     # maximum y value
sigma  = 0.0   # standard deviation of noise
deg    = 5     # polynomial degree
name   = "poly_deg{:d}".format(deg)

# polynomial model
model = lambda X: Polynomial_Model(X,deg)

# regression analysis
x,y,f  = gen_noisy_Franke(N,sigma,x0,x1,y0,y1)
LinReg = LinearRegression(np.array([x,y]).T,f[:,None],"Franke")
LinReg.add_model(model,name)
LinReg.use_method("Ridge")
LinReg.run_analysis(alpha=0.0)
approx = LinReg.model[name]

# display results
X,Y,surf_Franke,surf_model = prepare_surfaces(N_surf,x0,x1,y0,y1,approx)
plot_surfaces(  X,Y,surf_Franke,surf_model,LinReg.dirpath,deg)
plot_difference(X,Y,surf_Franke,surf_model,LinReg.dirpath,deg)

