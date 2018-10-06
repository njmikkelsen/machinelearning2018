import itertools
import numpy as np, numpy.random as rand
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from LinearRegression import LinearRegression

class ScalarFieldPolynomialModel():
  """
  Polynomial Approximation of an N-Dimensonal Scalar Field (Surface). The polynomial model's
  parameters are determined via Linear Regression, the analysis is performed by LinearRegression().
  Input variables:
    X | matrix whose N columns are the independent variables
    y | column vector with the scalar field
  """
  def __init__(self,X,y,surface_name):
    self.LinReg = LinearRegression(X,y,surface_name,True)
    self.data   = {"OLS":[],"Ridge":[],"Lasso":[]}
  
  # creates the design matrix for an (X.shape[0])-dimensional polynomial model of degree deg using data set X
  # includes every possible polynomial term up-to and including degree deg
  @staticmethod  
  def polynomial_design_matrix(X,deg):
    X_            = np.c_[np.ones(X.shape[0])[:,None],X]
    sets          = [s for s in itertools.combinations_with_replacement(range(X_.shape[1]),deg)]
    design_matrix = np.ones((X.shape[0],len(sets)))
    for i in range(1,len(sets)):
      design_matrix[:,i] = np.prod([X_[:,sets[i]]],axis=2)
    return design_matrix

  # main functionality
  def __call__(self,method="OLS",alpha=0,technique="",K=None):
    self.update_LinReg_models()
    def perform_run(method,alpha,technique,K):
      self.LinReg.use_method(method)
      self.LinReg.setup_resampling(technique,K)
      self.LinReg.run_analysis(alpha)
      self.data[method].append([[key,alpha,self.LinReg.model[key].beta] for key in self.LinReg.model.keys()])
    methods = ["OLS","Ridge","Lasso"] if method == "all" else [method]
    for method_ in methods: perform_run(method_,alpha,technique,K)
  
  # adds missing polynomials to self.LinReg
  def update_LinReg_models(self):
    for poly in self.polynomials.keys():
      if not poly in self.LinReg.model.keys():
        self.LinReg.add_model(self.polynomials[poly],poly)
    for model in self.LinReg.model.keys():
      if not model in self.polynomials.keys():
        self.LinReg.remove_model(model)
  
  # add polynomial(s) to the analysis
  def add_polynomial(self, deg):
    if hasattr(deg,"__len__"):
      for d in deg:
        if not d in self.polynomials.keys(): self.polynomials[d]   = lambda X: self.polynomial_design_matrix(X,d)
    else:
      if not deg in self.polynomials.keys(): self.polynomials[deg] = lambda X: self.polynomial_design_matrix(X,deg)
  
  # remove polynomial(s) from the analysis
  def remove_polynomial(self, deg):
    if hasattr(deg,"__len__"):
      for d in deg:
        if d in self.polynomials.keys(): del self.polynomials[d]
    else:
      if deg in self.polynomials.keys(): del self.polynomials[deg]
  
  # 

class FrankeApproximation(ScalarFieldPolynomialModel):
  """
  Polynomial Regression Analysis of Franke's Function:
    Franke = (3/4) g1 + (3/4) g2 + (1/2) g3 - (1/5) g4
  where
    g1 = exp( - (1/ 4)(9x-2)^2 - (1/ 4)(9y-2)^2 )
    g2 = exp( - (1/49)(9x+1)^2 - (1/10)(9y+1)^2 )
    g3 = exp( - (1/ 4)(9x-7)^2 - (1/ 4)(9y-3)^2 )
    g4 = exp( - (1/ 1)(9x-4)^2 - (1/ 1)(9y-7)^2 )
  This class is built on the PolynomialSurfaceModel() class.
  """
  def __init__(self,N,sigma=0,x0=0,x1=1,y0=0,y1=1,N_plot=400):
    self.N,self.sigma,self.x0,self.x1,self.y0,self.y1,self.N_plot = N,sigma,x0,x1,y0,y1,N_plot
    self.polynomials     = {5:lambda X:self.polynomial_design_matrix(X,5)}
    self.x,self.y,self.f = self.gen_noisy_Franke(N,sigma,x0,x1,y0,y1)
    self.X,self.Y        = np.meshgrid(np.linspace(x0,x1,N_plot),np.linspace(y0,y1,N_plot))
    self.F               = self.eval(self.X,self.Y)
    super().__init__(np.array([self.x,self.y]).T,self.f,"FrankeApprox")
  
  # Creates a noisy data set using Franke's function.
  @staticmethod
  def gen_noisy_Franke(N, sigma, x0=0, x1=1, y0=0, y1=1):
    x,y = rand.uniform(x0,x1,N),rand.uniform(y0,y1,N)
    return x, y, FrankeApproximation.eval(x,y) + sigma*rand.randn(N)
  
  # compute the exact Franke's function value
  @staticmethod
  def eval(x,y):
    g1 = np.exp(-0.25*((9*x-2)**2+(9*y-2)**2))
    g2 = np.exp(-(9*x+1)**2/49.-(9*y+1)**2/10.)
    g3 = np.exp(-0.25*((9*x-7)**2+(9*y-3)**2))
    g4 = np.exp(-(9*x-4)**2-(9*y-7)**2)
    return 0.75*g1 + 0.75*g2 + 0.50*g3 - 0.20*g3
  
  @staticmethod
  def ordinal(n):  
    """
    This was shamelessly stolen with love from a nice Mr. Ben Davis on Stack Exchange.
    The function prints the correct abbreviation for ordinal numbers. Ex: 1st, 2nd, 3rd, 4th.
    Reference: https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
    """
    return "%d%s" % (n,"tsnrhtdd"[(np.floor(n/10)%10!=1)*(n%10<4)*n%10::4])
  
  # display the exact Franke surface
  def plot_exact(self):
    fig = plt.figure(figsize=(14,8)); ax = fig.add_subplot(111,projection="3d")
    ax.plot_surface(self.X,self.Y,self.F,cmap=cm.Spectral,lw=0)
    ax.set_title("Surface Plot of Franke's Function",fontsize=22)
    ax.set_xlabel('x',fontsize=20); ax.set_ylabel('y',fontsize=20)
    plt.savefig(self.LinReg.dirpath+"exact_surface.png")
    plt.show()
  
  # build surface for plot_model() and plot_diff()
  def build_model_surface(self,method,idx,deg):
    if deg is None: l = [key for key in self.polynomials.keys()]; deg = l[0]
    for element in self.data[method][idx]:
      if deg == element[0]: penalty,beta = element[1],element[2]
    surf,x,f = np.zeros(self.X.shape),self.X[0],lambda x,y: self.polynomial_design_matrix(np.array([x,y]).T,deg)
    for i in range(self.N_plot):
      surf[i] = (f(x,self.Y[i,:])@beta).flatten()
    return surf, penalty, deg
  
  # display approximated Franke surface
  def plot_model(self,method="OLS",idx=0,deg=None):
    surf,penalty,deg = self.build_model_surface(method,idx,deg)
    fig = plt.figure(figsize=(14,8)); ax = fig.add_subplot(111,projection="3d")
    ax.plot_surface(self.X,self.Y,surf,cmap=cm.Spectral_r,lw=0)
    extra = "" if method == "OLS" else r" Using $\lambda =$ " + str(penalty)
    ax.set_title("{:s} Degree Polynomial Approximation of Franke's Function\n".format(self.ordinal(deg)) \
               + "With Model Parameters Determined by {:s} Regression{:s}".format(method,extra),fontsize=22)
    ax.set_xlabel("x",fontsize=20); ax.set_ylabel("y",fontsize=20)
    plt.savefig(self.LinReg.dirpath+"{:s}_surface_deg{:d}.png".format(method,deg))
    plt.show()
 
  # display difference of approximated Franke surface and exact Franke surface as a contour plot
  def plot_diff(self,method="OLS",idx=0,deg=None):
    surf,penalty,deg = self.build_model_surface(method,idx,deg)
    fig = plt.figure(figsize=(15,8)); ax = fig.add_subplot(111)
    cont = ax.contourf(self.X,self.Y,self.F-surf,60,cmap=cm.jet)
    plt.colorbar(cont)
    extra = "" if method == "OLS" else r" Using $\lambda =$ " + str(penalty)
    ax.set_title("Difference Between {:s} Degree Polynomal Approximation and Franke's Function\n".format(self.ordinal(deg)) +
                 "With Model Parameters Determined by {:s} Regression{:s}".format(method,extra),fontsize=22)
    ax.set_xlabel("x",fontsize=20); ax.set_ylabel("y",fontsize=20)
    plt.savefig(self.LinReg.dirpath+"{:s}_diff_deg{:d}.png".format(method,deg))
    plt.show()
  
