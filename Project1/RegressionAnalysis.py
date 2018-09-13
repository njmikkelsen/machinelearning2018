import numpy as np
import numpy.linalg as la
import numpy.random as rand

class RegressionAnalysis(object):
  """
  Parent class for:
    - PolynomialRegression
    - 
  """
  def __init__(self, x=None, y=None, dim=None):
    self.check_xy(x,y,dim)     # verify & save data input
    self.n_data = len(self.x)  # no. of data points
    self.n_beta = int(dim)     # no. of beta_i parameters
  
  # verify initialisation arguments
  def check_xy(self, x, y, dim):
    if sum([type(k).__module__ == np.__name__ for k in [x,y]]) != 2:
      raise TypeError("x and y must be NumPy ndarrays!")
    if x.size != y.size:
      raise ValueError("Unequal number of x and y values!")
    self.x = x.flatten()
    self.y = y.flatten()
  
  # computes the Mean Square Error
  def MSE(self, y_):
      return np.sum(np.square(self.y-y_))/float(self.n_data)
  
  # comptues the R2-score
  def R2(self, y_):
    ybar = np.sum(self.y)/float(self.n_data)
    return 1 - np.sum(np.square(self.y-y_))/np.sum(np.square(self.y-ybar))
    
class PolynomialRegression(RegressionAnalysis):
  """
  Subclass of RegressionAnalysis, automates a polynomial regression
  analysis of the given data set.
  """
  def __init__(self, x=None, y=None, degree=1):
    super().__init__(x, y, dim=degree+1)
    self.design_matrix()
    
  # creates the design matrix
  def design_matrix(self):
    self.X      = np.zeros((self.n_data,self.n_beta))
    self.X[:,0] = np.ones(self.n_data)
    for i in range(1,self.n_beta):
      self.X[:,i] = np.power(self.x,i)
  
    



if __name__ == "__main__":
  x = np.linspace(0,10,100)
  y = rand.randn(100)
  test = PolynomialRegression(x,y,2)
  y1 = y + 0.05*rand.randn(100)
  print("MSE =", test.MSE(y1))
  print("R2  =", test.R2(y1))

  
