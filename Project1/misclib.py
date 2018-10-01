import itertools
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def Franke(x,y):
  """
  Evaluates Franke's function. x and y can be floats, arrays or meshgrids.
  """
  g1 = np.exp(-0.25*((9*x-2)**2+(9*y-2)**2))
  g2 = np.exp(-(9*x+1)**2/49.-(9*y+1)**2/10.)
  g3 = np.exp(-0.25*((9*x-7)**2+(9*y-3)**2))
  g4 = np.exp(-(9*x-4)**2-(9*y-7)**2)
  return 0.75*g1 + 0.75*g2 + 0.50*g3 - 0.20*g3

def gen_noisy_Franke(N, s_noise, x0=0, x1=1, y0=0, y1=1):
  """
  Creates a noisy data set using Franke's function.
  """
  x     = rand.uniform(x0,x1,N)
  y     = rand.uniform(y0,y1,N)
  noise = s_noise*rand.randn(N)
  return x, y, Franke(x,y) + noise

def Polynomial_Model(X,deg):
  """
  Returns the design matrix for a deg-dimensional polynomial based on the input data X.
  Does not support polynomial expansions with 'missing' terms. Ex: f = a1 + a2*x + a3*x^3.
  """
  X_            = np.c_[np.ones(X.shape[0])[:,None],X]
  sets          = [s for s in itertools.combinations_with_replacement(range(X.shape[1]+1),deg)]
  design_matrix = np.ones((X.shape[0],len(sets)))
  for i in range(1,len(sets)):
    design_matrix[:,i] = np.prod([X_[:,sets[i]]],axis=2)
  return design_matrix

def prepare_surfaces(N_surf,x0,x1,y0,y1,poly_model):
  """
  Creates meshgrids meant for plotting with plot_surfaces().
  """
  X,Y   = np.linspace(x0,x1,N_surf),np.linspace(y0,y1,N_surf)
  XX,YY = np.meshgrid(X,Y)
  surf_Franke = Franke(XX,YY)
  surf_model  = np.zeros((XX.shape))
  for i in range(N_surf):
    surf_model[i] = poly_model(np.array([X,Y[i]*np.ones(N_surf)]).T).flatten()
  return XX,YY,surf_Franke,surf_model

"""
This was shamelessly stolen with love from a nice Mr. Ben Davis on Stack Exchange.
The function prints the correct abbreviation for ordinal numbers.
https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
"""
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(np.floor(n/10)%10!=1)*(n%10<4)*n%10::4])

def plot_surfaces(X,Y,surf_Franke,surf_model,dirpath,deg):
  """
  Plots and saves the surface plots of Franke's function and the polynomial approximation.
  """
  fig = plt.figure(figsize=(14,10))

  ax1 = fig.add_subplot(211,projection="3d")
  ax1.plot_surface(X,Y,surf_Franke,cmap=cm.Spectral_r,lw=0)
  ax1.set_title("Franke's function",fontsize=22)
  ax1.set_xlabel("x",fontsize=20)
  ax1.set_ylabel("y",fontsize=20)

  ax2 = fig.add_subplot(212,projection="3d")
  ax2.plot_surface(X,Y,surf_model,cmap=cm.Spectral_r,lw=0)
  ax2.set_title("{:s} order polynomial approximation of Franke's function".format(ordinal(deg)),fontsize=22)
  ax2.set_xlabel("x",fontsize=20)
  ax2.set_ylabel("y",fontsize=20)

  plt.tight_layout()
  plt.savefig(dirpath+"surf_plot_deg{:d}.png".format(deg))
  plt.show()

def plot_difference(X,Y,surf_Franke,surf_model,dirpath,deg):
  """
  Plots and saves the difference between Franke's function and the polynomial approximation as a contour plot.
  """
  fig     = plt.figure(figsize=(10,8))
  ax      = fig.add_subplot(111)
  contour = ax.contourf(X,Y,surf_Franke-surf_model,60,cmap=cm.jet)
  plt.colorbar(contour)
  ax.set_title("Difference between Franke's function\n" +
               "and a {:s} order polynomial approximation".format(ordinal(deg)),fontsize=22)
  ax.set_xlabel("x",fontsize=20)
  ax.set_ylabel("y",fontsize=20)
  plt.savefig(dirpath+"diff_plot_deg{:d}.png".format(deg))
  plt.show()



