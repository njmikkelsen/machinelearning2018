import sys, os
from datetime import datetime
import numpy as np
import numpy.linalg as la
import numpy.random as rand
import matplotlib.pyplot as plt

class LinearRegression(object):
  """
  The purpose of this class is to give the user a quick and easy way to perform
  several linear regressional analyses using different models.
  Three varieties of linear regression has been implemented:
  Ordinary Least Squares (OLS), Ridge Regression and Lasso Regression.
  
  Accessible to the user:
  
  Attributes:
    n_data  | Number of data points.
    n_beta  | Current number of parameters.
    y       | Dependent input data.
    y_      | Current prediction.
    X       | Current design matrix.
    beta    | Current parameters.
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
      raise TypeError("n_beta must be an integer greater than 0!")
    elif n_beta < 0:
      raise TypeError("n_beta must be greater than 0!")
    self.n_beta, self.svd = n_beta, False
  
  # build a design matrix for a univariate polynomial model
  def polynomial_model(self, x0, x1, degree):
    self.set_n_beta(degree+1)
    x           = np.linspace(x0,x1,self.n_data)
    self.X      = np.zeros((self.n_data,self.n_beta))
    self.X[:,0] = np.ones(self.n_data)
    for i in range(1,degree+1):
      self.X[:,i] = x**i
  
  # perform the regression analysis
  def analyse(self, method="OLS", lam=0, comment=''):
    if not method in ["OLS","Ridge","Lasso"]:
      raise ValueError("Invalid regression method!")
    # Singular Value Decomposition
    if not self.svd and method in ["OLS","Ridge"]:
      [self.U,self.d,self.VT],self.svd = la.svd(self.X,full_matrices=False), True
    # analyse parameters & compute prediction
    if   method == "OLS":
      self.beta = self.VT.T@np.diag(1/self.d)@self.U.T@self.y
    elif method == "Ridge":
      self.beta = self.VT.T@np.diag(self.d/(self.d**2+lam))@self.U.T@self.y
    elif method == "Lasso":
      print("syke")
    self.y_ = self.X@self.beta
    # compute confidence intervals
    sigma = np.sqrt(np.sum((self.y-self.y_)**2)/(self.n_data-self.n_beta))
    if   method == "OLS":
      D_matrix = np.diag(1/self.d**2)
    elif method == "Ridge":
      D_matrix = np.diag(self.d/((self.d**2+lam)**2))
    elif method == "Lasso":
      print("syke")
    self.R_beta = sigma*np.sqrt(np.abs(np.ravel(self.VT.T@D_matrix@self.VT)))
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
    n_records     = len(os.listdir("history"))+1
    self.filename = "history/record{:d}.dat".format(n_records)
    t             = datetime.now()
    with open(self.filename,"w") as h:
      h.write("LINEAR REGRESSION RECORD\n")
      h.write("time of analysis: {:d}/{:02d}/{:02d} {:02d}:{:02d}:{:02d}\n\n".format(\
              t.year,t.month,t.day,t.hour,t.minute,t.second))
      h.write('-'*130+"\n\noriginal data:\n\n")
      h.write(' '.join(["{:+17.10e}".format(self.y[i]) for i in range(self.n_data)]))
      h.write("\n")
      
  # write regression results to file
  def write_history(self, method, comment):
    with open(self.filename, "a") as h:
      h.write("\n" + '-'*130 + "\n\n")
      h.write("Method #{:d}\n\n".format(self.n_analyses))
      if comment != "":
        h.write("Comment:\n  " + comment + "\n")
      h.write("Regression method: {:s}\n".format(method))
      h.write("number of model parameters: {:d}\n".format(self.n_beta))
      h.write("MSE = {:+17.10e}\n".format(self.MSE))
      h.write("R2  = {:+17.10e}\n".format(self.R2))
      h.write("\noptimal parameters:\n\n")
      h.write(' '.join(["{:+17.10e}".format(self.beta[l]) for l in range(self.n_beta)]))
      h.write("\n\nradius of parameter confidence intervals:\n\n")
      h.write(' '.join(["{:+17.10e}".format(self.R_beta[l]) for l in range(self.n_beta)]))
      h.write("\n\nprediction:\n\n")
      h.write(' '.join(["{:+17.10e}".format(self.y_[k]) for k in range(self.n_data)]))
      h.write("\n\n")
      h.write("design matrix:\n\n")
      for k in range(self.n_data):
        h.write(' '.join(["{:+17.10e}".format(self.X[k,l]) for l in range(self.n_beta)]))
        h.write("\n")

# example of usage
if __name__ == "__main__":
  # setup
  N1, N2 = 50, 5
  x      = np.linspace(0,4,N1)
  y_     = np.cos(x) + 0.4*rand.randn(N1)
  LinReg = LinearRegression(y=y_)
  # compute and plot results
  plt.scatter(x,y_)
  plt.plot(x,np.cos(x),label="sine curve")
  for i in range(N2):
    LinReg.polynomial_model(np.min(x),np.max(x),i)
    LinReg.analyse("OLS")
    plt.plot(x,LinReg.y_,label="deg = {:d}".format(i))
  plt.title("Polynomial regression of a sine curve")
  plt.xlabel('x'); plt.ylabel('y')
  plt.legend(loc='best')
  plt.show()
  
