import numpy as np
from LinearRegression import *
from misclib import *

# Franke parameters
N      = 800    # number of data points
N_surf = 400   # number of surface grid points
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
LinReg = LinearRegression(np.array([x,y]).T,f[:,None],"Franke",False)
LinReg.dirpath = ""
LinReg.add_model(model,name)
"""
# OLS Regression
LinReg.use_method("OLS")
LinReg.run_analysis()
print(LinReg.model[name].beta.T)

X,Y,surf_Franke,surf_model = prepare_surfaces(N_surf,x0,x1,y0,y1,LinReg.model[name])
plot_surfaces(X,Y,surf_Franke,surf_model,LinReg.dirpath+"_OLS",deg)
"""
# Lasso Regression
LinReg.use_method("Lasso")
LinReg.run_analysis(alpha=1e-5)
print(LinReg.model[name].beta.T)

X,Y,surf_Franke,surf_model = prepare_surfaces(N_surf,x0,x1,y0,y1,LinReg.model[name])
plot_surfaces(X,Y,surf_Franke,surf_model,LinReg.dirpath+"_Lasso",deg)




#plot_difference(X,Y,surf_Franke,surf_model,LinReg.dirpath,deg)

