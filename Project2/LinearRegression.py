import numpy as np  
from sklearn.linear_model import Lasso
from sklearn.utils import resample
from misclib import *

class LinearRegressor(object):
  """
  Parent class for Univariate Linear regressor objects:
  OLSRegressor, RidgeRegressor, LassoRegressor
  -------------------------------------------------------
  """
  def __init__(self,std=True):
    self.N_inputs    = 0      # number of predictors/#columns in design matrix
    self.N_samples   = 0      # number of samples/#rows in design matrix
    self.X           = 0      # regression inputs/design matrix
    self.Y           = 0      # regression targets/outputs
    self.alpha       = 0      # regularization parameter
    self.standardize = std    # standardize data in preprocessing (remove mean and std deviation)
    self.regressed   = False  # regression completed
    self.compute_SVD = True   # see self.enter_SVD()
  
  # preprocessing regression variables X and Y
  def preprocess_data(self):
    if self.standardize:
      # compute statistics
      self.X_mean    = np.mean(self.X,axis=0)
      self.X_var     = np.var(self.X, axis=0)
      self.intercept = np.mean(self.Y)
      # adjust data
      self.X = (self.X - self.X_mean) / np.sqrt(self.X_var)
      self.Y = self.Y - self.intercept
  
  # regress X on Y
  def fit(self,X,Y,alpha=0):
    # save regression inputs and outputs
    self.N_inputs  = X.shape[0]
    self.N_samples = X.shape[1]
    self.X         = X.astype(np.float_)
    self.Y         = Y.astype(np.float_) if len(Y.shape)==2 else Y[:,None].astype(np.float_)
    self.alpha     = alpha
    # pre-process data for regression (center & normalize)
    self.preprocess_data()
    # compute regression coefficients (native to OLS, Ridge or Lasso)
    self.compute_coeff()
    self.regressed = True
  
  # use regression coefficients to predict outcome
  def predict(self,X,ravel_output=True):
    if self.regressed:
      prediction = (X)@(self.coeff) 
      if self.standardize: prediction += self.intercept
      return np.ravel(prediction) if ravel_output else prediction
    else: return None
  
  def enter_SVD(self,U,s,Vh):
    """
    Enter a precalculated Singular Value Decomposition of the design matrix before
    running fit(). This function's implementation stems from the desire to avoid
    having to decompose the design matrix twice in the same program.
    """
    self.U,self.s,self.Vh = U,s,Vh
    self.compute_SVD      = False

class OLSRegressor(LinearRegressor):
  """
  Linear Regression via Ordinary Least Squares
  -------------------------------------------------------------------------------
  The regression coefficients are computed using the SVD of the design matrix in
  order to avoid difficulties with singular matrices.
  """
  def __init__(self,std=True):
    self.regressor_type = "OLS"
    super().__init__(std)
    
  # regress X on Y
  def compute_coeff(self):
    # Singular Value Decomposition
    if self.compute_SVD:
      self.U,self.s,self.Vh = SVD(self.X)
    # compute coeff vector
    # self.coeff = (self.Vh.T) @ (np.diag(1./self.s)) @ (self.U.T) @ (self.Y)        # TODO: Why doesn't this work
    self.coeff = np.linalg.lstsq(self.X,self.Y,rcond=None)[0]
  
class RidgeRegressor(LinearRegressor):
  """
  Linear Regression via Ridge Regression (OLS Regression w/ L2 penalty)
  -------------------------------------------------------------------------------
  The regression coefficients are computed using the SVD of the design matrix in
  order to avoid difficulties with singular matrices.
  """
  def __init__(self,std=True):
    self.regressor_type = "Ridge"
    super().__init__(std)
  
  # regress X on Y
  def compute_coeff(self):
    # Singular Value Decomposition
    if self.compute_SVD:
      self.U,self.s,self.Vh = SVD(self.X)
    # compute coeff vector
    self.coeff = (self.Vh.T) @ (np.diag(self.s/(self.s**2+self.alpha))) @ (self.U.T) @ (self.Y)

class LassoRegressor(LinearRegressor):
  """
  Linear Regression via Lasso Regression (OLS Regression w/ L1 penalty)
  -------------------------------------------------------------------------------
  This class is a wrapper for scikit-learn's "sklearn.linear_model.Lasso" class.
  """
  def __init__(self,std=True,max_iter=10000):
    self.regressor_type = "Lasso"
    self.sklearn_Lasso  = Lasso(fit_intercept=std,normalize=std,max_iter=max_iter)
    super().__init__(std)
    self.std = False  # let scikit-learn handle preprocessing
  
  # regress X on Y
  def compute_coeff(self):
    self.sklearn_Lasso.set_params(alpha=self.alpha)
    self.sklearn_Lasso.fit(self.X,self.Y)
    self.coeff = self.sklearn_Lasso.coef_

class BootstrapResampler(object):
  """
  Bootstrap Resampler for Linear Regressor objects
  --------------------------------------------------------------
  Evaluates the boostrap estimate for model bias and variance.
  """
  def __init__(self,B):
    self.B      = B
    
  def run_resampling(self,Regressor,X,Y,lmbda):
    svd_loaded = not Regressor.compute_SVD
    # original regression
    Regressor.fit(X,Y,alpha=lmbda)
    prediction = Regressor.predict(X).flatten()
    MSE_       = MSE(Y.flatten(),prediction)
    # bootstrapping
    MSE_boot = np.zeros(self.B)
    if svd_loaded: Regressor.compute_SVD = True
    for b in range(self.B):
      # Bootstrap sample
      X_boot,Y_boot = resample(X,Y)
      # regression on bootstrap sample
      Regressor.fit(X_boot,Y_boot,alpha=lmbda)
      prediction  = Regressor.predict(X_boot).flatten()
      MSE_boot[b] = MSE(Y_boot,prediction)
    # compute contribution to bias and variance
    Bias2,Var = Bootstrap_Bias_Variance_computation(MSE_boot,MSE_)
    if svd_loaded: Regressor.compute_SVD = False
    return MSE_,Bias2,Var

# jitted computation in run_sampling() in BootstrapResamper class
@jit(nopython=True,parallel=True,fastmath=True)
def Bootstrap_Bias_Variance_computation(MSE_boot,MSE_):
  Bias = np.mean(MSE_boot-MSE_)
  Var  = np.mean((MSE_boot-np.mean(MSE_boot))**2)
  return Bias**2,Var

    
    
