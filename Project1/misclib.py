import itertools
from imageio import imread
import numpy as np, numpy.random as rand
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from LinearRegression import LinearRegression

class ScalarField_PolynomialApproximation():
  """
  Polynomial Approximation of an N-Dimensional Scalar Field (Surface). The polynomial parameters
  are determined via Linear Regression Analysis, which is performed by the LinearRegression() class.
  Input variables:
    X | matrix whose N columns are the independent variables
    y | column vector with the scalar field
  """
  def __init__(self,X,y,ScalarField):
    self.LinReg      = LinearRegression(X,y,ScalarField,True)
    self.P,self.data = [],{"OLS":{},"Ridge":{},"Lasso":{}}
  
  # creates the design matrix for an (X.shape[0])-dimensional polynomial model of degree deg using data set X
  # includes every possible polynomial term up-to and including degree deg
  @staticmethod  
  def design_matrix(X,deg):
    X_            = np.c_[np.ones(X.shape[0])[:,None],X]
    sets          = [s for s in itertools.combinations_with_replacement(range(X_.shape[1]),deg)]
    design_matrix = np.ones((X.shape[0],len(sets)))
    for i in range(1,len(sets)):
      design_matrix[:,i] = np.prod([X_[:,sets[i]]],axis=2)
    return design_matrix
  
  # main functionality - determine polynomial coefficients and save results for all polynomials
  def __call__(self,method="OLS",alpha=0,technique="",K=1):
    if len(self.P) == 0: raise ValueError("No polynomial has been added")
    self.update_LinReg()
    self.update_data_keys()
    self.determine_coefficients(method,alpha,technique,K)
  
  # adds missing and removes obsolete polynomials to self.LinReg
  def update_LinReg(self):
    current_models = sorted([key for key in self.LinReg.model.keys()])
    for deg in self.P:
      if not deg in current_models:
        F = lambda X: self.design_matrix(X,deg)
        self.LinReg.add_model(F,deg)
    for model in current_models:
      if not model in self.P:
        self.LinReg.remove_model(model)
  
  # update data keys
  def update_data_keys(self):
    for method in self.data.keys():
      for deg in self.P:
        if not deg in self.data[method].keys():
          self.data[method][deg] = {}

  # run regression analysis
  def determine_coefficients(self,method,alpha,technique,K):
    self.LinReg.use_method(method)
    self.LinReg.run_analysis(alpha,technique,K)
    for deg in self.P:
      idx = len(self.data[method][deg].keys())
      model = self.LinReg.model[deg]
      self.data[method][deg][idx] = [alpha,model.MSE,model.R2,model.beta]
  
  # add polynomial(s) to the analysis
  def add_polynomial(self, deg):
    if hasattr(deg,"__len__"):
      for d in deg:
        if not d in self.P: self.P.append(d)
    else:
      if not deg in self.P: self.P.append(deg)
    self.P.sort()
  
  # remove polynomial(s) from the analysis
  def remove_polynomial(self, deg):
    if hasattr(deg,"__len__"):
      for d in deg:
        if d in self.P: self.P.remove(d)
    else:
      if deg in self.P: self.P.remove(deg)
  

class Surface_PolynomialApproximation(ScalarField_PolynomialApproximation):
  """
  Polynomial approximation of a surface z = f(x,y).
  This is an expansion of the regular ScalarField_PolynomialApproximation class
  whose primary function is plotting capabilities.
  """
      
  @staticmethod
  def ordinal(n):  
    """
    This was shamelessly stolen with love from a nice Mr. Ben Davis on Stack Exchange.
    The function prints the correct abbreviation for ordinal numbers. Ex: 1st, 2nd, 3rd, 4th.
    Reference: https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
    """
    return "%d%s" % (n,"tsnrhtdd"[(np.floor(n/10)%10!=1)*(n%10<4)*n%10::4])
  
  # build surface for plot_model() and plot_contour()
  def build_surface(self,X,Y,method,deg,idx):
    beta   = self.data[method][deg][idx][3]
    z      = lambda x,y: self.design_matrix(np.array([x,y]).T,deg)
    surf,x = np.zeros(X.shape),X[0]
    for i in range(X.shape[0]):
      surf[i] = (z(x,Y[i,:])@beta).flatten()
    return surf
  
  # 3d plot of surface using meshgrid arrays X, Y and surf
  @staticmethod
  def plot_surface(X,Y,surf,title,fig_name,dirpath):
    fig = plt.figure(figsize=(14,8))
    ax  = fig.add_subplot(111,projection="3d")
    ax.plot_surface(X,Y,surf,cmap=cm.Spectral_r,lw=0)
    ax.set_title(title,fontsize=20)
    ax.set_xlabel("x",fontsize=20)
    ax.set_ylabel("y",fontsize=20)
    ax.set_zlabel("z",fontsize=20)
    plt.savefig(dirpath+"{:s}.png".format(fig_name))
    plt.show()
  
  # plot contour plot of surface from meshgrid arrays X, Y and surf
  @staticmethod
  def plot_contour(X,Y,contour,title,fig_name,dirpath):
    fig  = plt.figure(figsize=(15,8))
    ax   = fig.add_subplot(111)
    cont = ax.contourf(X,Y,contour,60,cmap=cm.jet)
    plt.colorbar(cont)
    ax.set_title(title,fontsize=20)
    ax.set_xlabel("x",fontsize=20)
    ax.set_ylabel("y",fontsize=20)
    plt.savefig(dirpath+"{:s}.png".format(fig_name))
    plt.show()

class Franke_PolynomialApproximation(Surface_PolynomialApproximation):
  """
  Polynomial approximation of Franke's Function via Linear Regression Analysis.
  Franke's function is a two-dimensional surface z = f(x,y) given by
    Franke = (3/4) g1 + (3/4) g2 + (1/2) g3 - (1/5) g4
  where
    g1 = exp( - (1/ 4)(9x-2)^2 - (1/ 4)(9y-2)^2 )
    g2 = exp( - (1/49)(9x+1)^2 - (1/10)(9y+1)^2 )
    g3 = exp( - (1/ 4)(9x-7)^2 - (1/ 4)(9y-3)^2 )
    g4 = exp( - (1/ 1)(9x-4)^2 - (1/ 1)(9y-7)^2 )
  """
  def __init__(self,N,sigma=0,x0=0,x1=1,y0=0,y1=1,N_plot=400):
    self.N,self.sigma,self.x0,self.x1,self.y0,self.y1,self.N_plot = N,sigma,x0,x1,y0,y1,N_plot
    self.x,self.y,self.f = self.generate_noisy_data(N,sigma,x0,x1,y0,y1)
    self.X,self.Y        = np.meshgrid(np.linspace(x0,x1,N_plot),np.linspace(y0,y1,N_plot))
    self.F               = self.eval(self.X,self.Y)
    super().__init__(np.array([self.x,self.y]).T,self.f,"FrankeApprox")
  
  # Creates a noisy data set using Franke's function.
  @staticmethod
  def generate_noisy_data(N, sigma, x0=0, x1=1, y0=0, y1=1):
    x,y = rand.uniform(x0,x1,N),rand.uniform(y0,y1,N)
    return x, y, Franke_PolynomialApproximation.eval(x,y) + sigma*rand.randn(N)
  
  # compute Franke's function
  @staticmethod
  def eval(x,y):
    g1 = np.exp(-0.25*((9*x-2)**2+(9*y-2)**2))
    g2 = np.exp(-(9*x+1)**2/49.-(9*y+1)**2/10.)
    g3 = np.exp(-0.25*((9*x-7)**2+(9*y-3)**2))
    g4 = np.exp(-(9*x-4)**2-(9*y-7)**2)
    return 0.75*g1 + 0.75*g2 + 0.50*g3 - 0.20*g3
  
  # plot the exact Franke surface
  def plot_exact(self):
    self.plot_surface(self.X,self.Y,self.F,"Surface Plot of Franke's Function","exact_surface",self.LinReg.dirpath)
  
  # surface plot approximated Franke surface
  def plot_model(self,method="OLS",deg=None,idx=0):
    deg, title, surf = self.prepare_plot(method,deg,idx,"model")
    self.plot_surface(self.X,self.Y,surf,title,"surface_deg{:d}_{:s}".format(deg,method),self.LinReg.dirpath)

  # contour plot difference of approximated and exact Franke surface
  def plot_diff(self,method="OLS",deg=None,idx=0):
    deg, title, surf = self.prepare_plot(method,deg,idx,"diff")
    self.plot_contour(self.X,self.Y,surf-self.F,title,"diff_deg{:d}_{:s}".format(deg,method),self.LinReg.dirpath)

  # prepare for plot
  def prepare_plot(self,method,deg,idx,plot_type):
    if deg is None: deg = sorted([key for key in self.data[method].keys()])[0]
    penalty = self.data[method][deg][0]
    surf    = self.build_surface(self.X,self.Y,method,deg,idx)
    extra   = "" if method == "OLS" else r" Using $\lambda =$ " + str(penalty)
    title_  = ["Surface Plot of ","of"] if plot_type=="model" else ["Difference Between ","and"]
    title   = title_[0]+"{:s} Degree Polynomial Approximation ".format(self.ordinal(deg))+title_[1]+" Franke's Function\n" + \
              "With Coefficients Determined by {:s} Regresion{:s}".format(method,extra)
    surf    = self.build_surface(self.X,self.Y,method,deg,idx)
    return deg, title, surf

class Terrain_PolynomialApproximation(Surface_PolynomialApproximation):
  """
  Polynomial approximation of three-dimensional terrain-data stored in .tif files.
  The data is a surface z = f(x,y).
  """
  def __init__(self,filepath,nx=1,ny=1):
    self.filepath,terrain        = filepath,imread(filepath)
    self.terrain                 = np.asarray(terrain)[::ny,::nx]
    self.X,self.Y                = np.meshgrid(np.linspace(0,1,self.terrain.shape[1]),np.linspace(0,1,self.terrain.shape[0]))
    self.x,self.y,self.f         = [a.flatten() for a in [self.X,self.Y,self.terrain]]
    super().__init__(np.array([self.x,self.y]).T,self.f,"TerrainApprox")
  
  # plot terrain data
  def plot_terrain(self,contour=True,surface=False):
    if contour: self.plot_contour(self.X,self.Y,self.terrain,"Contour Plot of Terrain Data","terrain_contour",self.LinReg.dirpath)
    if surface: self.plot_surface(self.X,self.Y,self.terrain,"Surface Plot of Terrain Data","terrain_surface",self.LinReg.dirpath)
   
  # surface plot polynomial model
  def plot_model(self,method="OLS",deg=None,idx=0):
    deg, title, surf = self.prepare_plot(method,deg,idx,"model")
    self.plot_surface(self.X,self.Y,surf,title,"surface_deg{:d}_{:s}".format(deg,method),self.LinReg.dirpath)
  
  # contour plot difference between polynomial model and terrain data
  def plot_diff(self,method="OLS",deg=None,idx=0):
    deg, title, surf = self.prepare_plot(method,deg,idx,"diff")
    self.plot_contour(self.X,self.Y,surf-self.terrain,title,"diff_deg{:d}_{:s}".format(deg,method),self.LinReg.dirpath)
  
  # prepare for plot
  def prepare_plot(self,method,deg,idx,plot_type):
    if deg is None: deg = sorted([key for key in self.data[method].keys()])[0]
    penalty = self.data[method][deg][0]
    extra   = "" if method == "OLS" else r" Using $\lambda =$ " + str(penalty)
    title_  = ["Surface Plot of ","of"] if plot_type=="model" else ["Difference Between ","and"]
    title   = title_[0]+"{:s} Degree Polynomial Approximation ".format(self.ordinal(deg))+title_[1]+" Terrain Data\n" + \
              "With Coefficients Determined by {:s} Regresion{:s}".format(method,extra)
    surf    = self.build_surface(self.X,self.Y,method,deg,idx)
    return deg, title, surf

    
