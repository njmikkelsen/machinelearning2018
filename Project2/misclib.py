import sys
import numpy as np
from numba import jit
from scipy.linalg import svd
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score

# Singular Value Decomposition
def SVD(matrix):
  return svd(matrix,full_matrices=False)
  
# Mean Squared Error
def MSE(T,Y):
  return mean_squared_error(T,Y)

# Coefficient of Determination
def R2(T,Y):
  return r2_score(T,Y)

# labelling accuracy
def accuracy(T,Y):
  return accuracy_score(T,Y)

def scientific_number(x,d):
  """
  This function accepts a number x and returns its corresponding significand to d significant
  figurs and its base-10 exponent. Meant for printing scientific numbers in matplotlib.
  """
  X = ("{:.{:d}e}".format(x,d)).split("e")
  return X[0],"{:d}".format(int(X[1]))

class progress_bar(object):
  """
  A command-line progress bar for use in large loops (as a sanity check).
  -----------------------------------------------------------------------
  Argument N = number of loop iterations. Usage:

  BAR = misclib.progress_bar(N)
  for i in range(N):
    # do something
    BAR.update()
  """
  def __init__(self,N,add_space=False):
    self.n = 0
    self.N = N
    self.add_space = add_space
    if add_space: print('')
    sys.stdout.write('\r['+' '*20+']    0 % ')
  def update(self):
    self.n += 1
    if self.n < self.N:
      sys.stdout.write('\r[{:20s}] {:4.0f} % '.format('='*int(20*self.n/self.N),100.*self.n/self.N))
    else:
      sys.stdout.write('\r['+'='*20+']  100 % Done!\n')
      if self.add_space: print('')





