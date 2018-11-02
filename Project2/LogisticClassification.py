import numpy as np
from misclib import progress_bar

class LogisticClassifier(object):
  """
  Classification via Logistic Regression & Standard Gradient Descent
  --------------------------------------------------------------------
  """
  def __init__(self,std_beta=1): 
    self.std_beta  = 1.     # standard deviation in initial beta coefficients
    self.regressed = False  # indicates whether classifier is fitted
  
  def fit(self,N,X,Y,gamma=1e-2):
    self.beta = self.std_beta*np.random.randn(X.shape[1]).astype(np.float_)
    progress  = progress_bar(N)
    for i in range(N):
      prob       = 1./(1.+np.exp(-(X) @ (beta0)))
      self.beta += gamma*((XT)@(Y-prob))
      progress.update()
    self.regressed = True
  
  def predict(self,X):
    if self.regressed:
      prediction = (((X) @ (self.beta))>0).astype(np.int8)
      prediction[np.where(prediction)==0] = -1
      return prediction
    else: return None
  
  
  
