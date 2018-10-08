import sys
import numpy as np
from LinearRegression import *
from misclib import *

# numpy random seed
np.random.seed(1996)

# Franke parameters
N     = 400   # number of data points
x0    = 0     # minimum x value
x1    = 1     # maximum x value
y0    = 0     # minimum y value
y1    = 1     # maximum y value
sigma = 0.0   # standard deviation of noise

# regression parameters
deg    = 5      # polynomial degree
N_lam  = 100    # number of penalty parameters
lam0   = 1e-9   # lower penalty
lam1   = 1.8e-1   # upper penalty
Boot   = True   # whether to include Bootstrap
K      = 200    # number of iterations in Bootstrap algorithm
OLS    = True   # whether to include OLS regression
Ridge  = True   # whether to include Ridge regression
Lasso  = True   # whether to include Lasso regression

# adjustments
Boot_ = "Bootstrap" if Boot else ""
N_lam = 1 if (OLS == True and Ridge == Lasso == False) else N_lam

# standard setup
A = Franke_PolynomialParametrisation(N,sigma,x0,x1,y0,y1,N)
A.plot_exact()
A.add_polynomial(deg)
f = open(A.LinReg.dirpath + "Franke.dat",'w')
f.write("Polynomial Parametrisation of Franke's Function Using Linear Regression\n\n")
f.write("Franke Parameters:")
f.write("""
  number of data points          = {:d}
  minimim x value                = {:+e}
  maximum x value                = {:+e}
  minimum y value                = {:+e}
  maximum y value                = {:+e}
  standard deviation of noise    = {:+e}\n\n""".format(N,x0,x1,y0,y1,sigma))
f.write("Regression parameters:")
f.write("""
  polynomial degree              = {:d}
  number of penalty parameters   = {:+e}
  minimum penalty                = {:+e}
  maximum penalty                = {:+e}
  include Bootstrap              = {:s}
  number of Bootstrap iterations = {:d}
  include OLS regression         = {:s}
  include Ridge regression       = {:s}
  include Lasso regression       = {:s}\n\n""".format(deg,N_lam,lam0,lam1,str(Boot),K,str(OLS),str(Ridge),str(Lasso)))
f.write('-'*70+"\n")

# perform fits
alpha = np.linspace(lam0,lam1,N_lam)
print("Fitting functions using Regression:")
for i in range(N_lam):
  sys.stdout.write('\r[{:20s}] {:4.0f} % '.format('='*int(20*i/N_lam),100.*i/N_lam))
  sys.stdout.flush()
  if OLS:   A("OLS",  alpha[i],Boot_,K)
  if Ridge: A("Ridge",alpha[i],Boot_,K)
  if Lasso: A("Lasso",alpha[i],Boot_,K)
print('\r['+'='*20+']  100 % Done!\n')

# plot error term dependence on the penalty
if Ridge or Lasso:
  A.plot_error_penalty_dependence(deg,Boot_,K)

# function for writing to file
def write_to_file(method):
  if   method == "OLS":   f.write("Ordinary Least Squares:\n")
  elif method == "Ridge": f.write("Ridge Regression:\n")
  else:                   f.write("Lasso Regression:\n")
  n = len(A.data[method][deg]) if method != "OLS" else 1
  for i in range(n):
    data = A.data[method][deg][i]
    if not method == "OLS": f.write("\n  penalty         = {:+e}\n".format(data[0]))
    else: f.write("\n")
    f.write("  MSE             = {:+e}\n".format(data[1]))
    f.write("  R2              = {:+e}\n".format(data[2]))
    f.write("  sum(std_beta^2) = {:+e}\n".format(np.sum(data[4])))
    if len(data) > 5:
      f.write("  using {:s} with K = {:d}:\n".format(data[9],data[10]))
      f.write("    avg(MSE) = {:+e}\n".format(data[5]))
      f.write("    avg(R2)  = {:+e}\n".format(data[6]))
      f.write("    Bias     = {:+e}\n".format(data[7]))
      f.write("    Variance = {:+e}\n".format(data[8]))
  f.write("\n"+'-'*70+"\n")

# write results to file
if OLS:   write_to_file("OLS")
if Ridge: write_to_file("Ridge")
if Lasso: write_to_file("Lasso")

# plot OLS fits
if OLS:
  A()
  A.plot_model(idx=N_lam+1)
  A.plot_diff(idx=N_lam+1)

f.close()

