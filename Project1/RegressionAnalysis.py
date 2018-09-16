import sys, os
from datetime import datetime
import numpy as np
import numpy.linalg as la
import numpy.random as rand
import matplotlib.pyplot as plt

class LinearRegression(object):
  """
  The purpose of this class is to give the user a quick and easy way to perform
  several linear regressional analyses using different models. Three varieties
  of linear regression has been implemented: Ordinary Least Squares (OLS),
  Ridge Regression and Lasso Regression.
  
  Accessible to the user:
  
  Attributes:
    y       | Dependent input data.
    n_data  | Number of data points.
    n_beta  | Current number of parameters.
    X       | Current design matrix.
    beta    | Current parameters.
    y_      | Current prediction.
    MSE     | Current mean squared error.
    R2      | Current R2-score.
  
  Methods:
    set_n_beta    | Adjust n_beta.
    analyse       | Perform the regression analysis.
    
  All results are automagically written to file under the "history" directory.
  """
  def __init__(self, y=None):
    # verify data input
    if type(y).__module__ != np.__name__:
      raise TypeError("Data array must be a flat NumPy ndarray!")
    self.y      = y.flatten()
    self.n_data = len(self.y)
    self.svd    = False
    self.setup_history()
  
  # set current n_beta
  def set_n_beta(self, n_beta):
    if not isinstance(n_beta,int):
      raise TypeError("n_beta must be an integer!")
    elif self.n_beta > 0:
      raise TypeError("n_beta must be greater than 0!")
    self.n_beta, self.svd = n_beta, False    
  
  def analyse(self, method="OLS", lam=0, comment=''):
    # Singular Value Decomposition
    if not self.svd:
      [self.U,self.d,self.VT],self.svd = la.svd(self.X,full_matrices=False), True
    # analyse parameters
    if   method == "OLS":
      self.beta = self.VT.T@np.diag(1/self.d)@self.U.T@self.y
    elif method == "Ridge":
      self.beta = self.VT.T@np.diag(self.d/(self.d**2+lam))@self.U.T@self.y
    elif method == "Lasso":
      print("syke")
    self.y_ = self.X@self.beta
    # compute error scores
    self.MSE = np.sum(np.square(self.y-self.y_))/float(self.n_data)
    ybar     = np.sum(self.y)/float(self.n_data)
    self.R2  = 1 - np.sum(np.square(self.y-self.y_))/np.sum(np.square(self.y-ybar))
    # write regression results to file
    self.n_analyses += 1
    self.write_history(method, comment)
  
  # setup the history file for recording the analyses
  def setup_history(self):
    self.n_analyses = 0
    if not os.path.isdir("history"):
      os.mkdir("history")
    n_records     = len(os.listdir("history"))
    self.filename = "history/record{:d}.dat".format(0)
    t             = datetime.now()
    with open(self.filename,"w") as h:
      h.write("LINEAR REGRESSION RECORD\n")
      h.write("time of analysis: {:d}/{:02d}/{:02d} {:02d}:{:02d}:{:02d}\n\n".format(\
              t.year,t.month,t.day,t.hour,t.minute,t.second))
      h.write('-'*130+"\n\noriginal data:\n\n")
      h.write(' '.join(["{:+17.10e}".format(self.y[i]) for i in range(self.n_data)]))
      
  # write regression results to file
  def write_history(self, method, comment):
    with open(self.filename, "a") as h:
      h.write("\n\n" + '-'*130 + "\n\n")
      h.write("Method #{:d}\n\n".format(self.n_analyses))
      if comment != "":
        h.write("Comment:\n  " + comment + "\n")
      h.write("Regression method: {:s}\n".format(method))
      h.write("number of model parameters: {:d}\n".format(self.n_beta))
      h.write("MSE = {:f}\n".format(self.MSE))
      h.write("R2  = {:f}\n".format(self.R2))
      h.write("\noptimal parameters:\n\n")
      h.write(' '.join(["{:+17.10e}".format(self.beta[l]) for l in range(self.n_beta)]))
      h.write("\n\n")
      h.write("prediction:\n\n")
      h.write(' '.join(["{:+17.10e}".format(self.y_[k]) for k in range(self.n_data)]))
      h.write("\n\n")
      h.write("design matrix:\n\n")
      for k in range(self.n_data):
        h.write(' '.join(["{:+17.10e}".format(self.X[k,l]) for l in range(self.n_beta)]))
        h.write("\n")


class PolynomialRegression(LinearRegression):
  """
  This is a subclass of LinearRegression whose purpose is to simplify the process
  of creating the design matrix for
  """
  def __init__(self, y=None, dim=1):
    if not isinstance(dim,int):
      raise TypeError("Polynomial dimensionality must be an integer!")
    if dim < 1:
      raise ValueError("Polynomial dimensionality must be at least 1!")
    self.dim = dim
    super().__init__(y)

  # creates the design matrix for polynomial regression
  def design_polynomial(self, x, degree=-1):
    if isinstance(degree,int):
      if degree > 0:
        self.n_beta = degree + 1
    self.X      = np.zeros((self.n_data,self.n_beta))
    self.X[:,0] = np.ones(self.n_data)
    for i in range(1,self.n_beta):
      self.X[:,i] = np.power(x,i)
    self.svd = False
  
  # adds a comment on polynomial regression
  def analyse(self, method="OLS", lam=0, comment=""):
    s = "Polynomial model with degree {:d}.\n".format(self.n_beta-1)
    if comment != '':
      s += "  " + comment + "\n"
    super().analyse(method, lam, comment = s)


if __name__ == "__main__":
  N    = 20
  x    = np.linspace(-0.5,1,N)
  y    = 2*x*x + 0.3*rand.randn(N)
  
  test = PolynomialRegression(y)
  test.design_polynomial(x=x, degree=2)

  test.analyse("OLS")
  y_1 = test.y_

  test.analyse("Ridge",0.1)
  y_2 = test.y_

  plt.title("Example: polynomial interpolation")
  plt.scatter(x,y)
  plt.plot(x,y_1,label="OLS",c='red')
  plt.plot(x,y_2,label="Ridge",c='blue')
  plt.legend(loc='best')
  plt.show()

