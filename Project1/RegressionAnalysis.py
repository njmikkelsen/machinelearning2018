import numpy as np
import numpy.linalg as la
import numpy.random as rand

class LinearRegression(object):
  """
  The purpose of this class is to give the user a quick and easy way to perform
  several linear regressional analyses using different models. Three varieties
  of linear regression has been implemented: Ordinary Least Squares (OLS),
  Ridge Regression and Lasso Regression.
  
  Attributes:
    x       | Independent input data.
    y       | Dependent input data.
    n_data  | Number of data points.
    n_beta  | Current number of parameters.
    X       | Current design matrix.
    beta    | Current parameters.
    y_      | Current prediction.
    history | Records the history of all solutions, errors, etc.
    MSE     | Current mean squared error.
    R2      | Current R2-score.
  
  Methods:
    check_xy                | Verify input data is of same size and type numpy.ndarray.
    check_n_beta            | Verify n_beta is an integer greater than 0.
    MSE                     | Computes the current MSE rating.
    R2                      | Computes the current R2-score.
    design_polynomial_model | Designs X according to polynomial regression.
    write_history           | Writes the contents of history attribute to file.
    print_history           | Prints the contents of history attribute to terminal.
  
  Class usage:
    1) declare LinearRegression instance with data set (x,y)
    2) for each analysis:
      a) adjust n_beta and X
      b) solve for parameters and compute prediction
      c) compute error/score
    3) write/print history
  """
  def __init__(self, x=None, y=None):
    self.check_xy(x,y)
    self.n_data  = len(self.x)
    self.n_beta  = 1
    self.X       = np.eye(self.n_data)
    self.beta    = self.y
    self.y_      = np.zeros(self.n_data)
    self.history = {"n_beta":[], "beta":[], "y_":[], "MSE":[], "R2":[]}
    self.MSE     = 0.
    self.R2      = 1.
  
  # verify initialisation arguments
  def check_xy(self, x, y):
    if sum([type(k).__module__ == np.__name__ for k in [x,y]]) != 2:
      raise TypeError("x and y must be NumPy ndarrays!")
    if x.size != y.size:
      raise ValueError("Unequal number of x and y values!")
    self.x = x.flatten()
    self.y = y.flatten()
    
  # define current n_beta
  def check_n_beta(self):
    if not isinstance(self.n_beta,int):
      raise TypeError("n_beta must be an integer!")
    elif self.n_beta > 0:
      raise TypeError("n_beta must be greater than 0!")
  
  # computes the Mean Square Error for the current solution
  def MSE(self):
      return np.sum(np.square(self.y-self.y_))/float(self.n_data)
  
  # comptues the R2-score for the current solution
  def R2(self):
    ybar = np.sum(self.y)/float(self.n_data)
    return 1 - np.sum(np.square(self.y-self.y_))/np.sum(np.square(self.y-ybar))
      
  # creates the design matrix for polynomial regression
  def design_polynomial(self, degree=-1):
    if isinstance(degree,int):
      if degree > 0:
        self.n_beta = degree
    self.X      = np.zeros((self.n_data,self.n_beta))
    self.X[:,0] = np.ones(self.n_data)
    for i in range(1,self.n_beta):
      self.X[:,i] = np.power(self.x,i)
  
  # write history to file
  def write_history(self):
    pass
    



if __name__ == "__main__":
  x = np.linspace(0,10,100)
  y = rand.randn(100)
  test = LinearRegression(x,y,2)
  y1 = y + 0.05*rand.randn(100)
  print("MSE =", test.MSE(y1))
  print("R2  =", test.R2(y1))

  
