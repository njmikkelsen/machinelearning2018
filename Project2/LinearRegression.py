import numpy as np
from scipy.linalg import svd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from misclib import progress_bar, computations

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
    self.standardize = std    # standardize data in preprocessing (remove mean and set std to unity)
    self.regressed   = False  # regression completed
    self.compute_SVD = True   # see self.enter_SVD()
  
  # preprocessing regression variables X and Y
  def preprocess_data(self):
    # compute statistics
    self.X_mean    = np.mean(self.X,axis=0) if self.standardize else np.zeros(self.X.shape[1])
    self.X_var     = np.var(self.X, axis=0) if self.standardize else np.ones(self.X.shape[1])
    self.intercept = np.mean(self.Y)        if self.standardize else 0
    # adjust data
    self.X = (self.X - self.X_mean) / np.sqrt(self.X_var)
    self.Y = self.Y - self.intercept
  
  # regress X on Y
  def fit(self,X,Y,alpha=0):
    # save regression inputs and outputs
    self.N_inputs  = X.shape[0]
    self.N_samples = X.shape[1]
    self.X         = X.astype(np.float_)
    self.Y         = Y.astype(np.float_)
    self.alpha     = alpha
    # pre-process data for regression (center & normalize)
    self.preprocess_data()
    # compute regression coefficients (native to OLS, Ridge or Lasso)
    self.compute_coeff()
    self.regressed = True
  
  # use regression coefficients to predict outcome
  def predict(self,X,ravel_output=True):
    if self.regressed:
      prediction = (X)@(self.coeff) + self.intercept
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
  # regress X on Y
  def compute_coeff(self):
    # Singular Value Decomposition
    if self.compute_SVD:
      self.U,self.s,self.V = LinearRegressor.SVD(self.X)
    # compute coeff vector
    self.coeff = (self.Vh.T) @ (np.diag(1./self.s)) @ (self.U.T) @ (self.Y)

class RidgeRegressor(LinearRegressor):
  """
  Linear Regression via Ridge Regression (OLS Regression w/ L2 penalty)
  -------------------------------------------------------------------------------
  The regression coefficients are computed using the SVD of the design matrix in
  order to avoid difficulties with singular matrices.
  """
  # regress X on Y
  def compute_coeff(self):
    # Singular Value Decomposition
    if self.compute_SVD:
      self.U,self.s,self.V = LinearRegressor.SVD(self.X)
    # compute coeff vector
    self.coeff = (self.Vh.T) @ (np.diag(self.s/(self.s**2+self.alpha))) @ (self.U.T) @ (self.Y)

class LassoRegressor(LinearRegressor):
  """
  Linear Regression via Lasso Regression (OLS Regression w/ L1 penalty)
  -------------------------------------------------------------------------------
  This class is a wrapper for scikit-learn's "sklearn.linear_model.Lasso" class.
  """
  def __init__(self,std=True):
    self.sklearn_Lasso = Lasso(fit_intercept=std,normalize=std)
    super().__init__(std)
    self.std = False  # let scikit-learn handle preprocessing
  
  # regress X on Y
  def compute_coeff(self):
    self.sklearn_Lasso.set_params(self.alpha)
    self.sklearn_Lasso.fit()
    self.coeff = self.sklearn_Lasso.coef_

class JackknifeResampler(object):
  """
  Jackknife Resampler for Linear Regressor objects
  --------------------------------------------------
  """
  def __init__(self,Regressor,B):
    self.LinReg      = Regressor
    self.B           = Regressor.N_samples-1
    
  def run_resampling(self,X,Y,Lambda):
    self.MSE  = np.zeros(X.shape[0])
    self.R2   = np.zeros(X.shape[0])
    self.Bias = np.zeros(X.shape[0])
    self.Var  = np.zeros(X.shape[0])
    print("running Jackknife resampling...")
    progress = progress_bar(self.B*len(Lambda))
    for i,lmbda in enumerate(Lambda):
      # initial regression
      self.LinReg.fit(X,Y,alpha=lmbda)
      prediction  = self.LinReg.predict(X)
      self.MSE[i] = computations.metrics.MSE(Y,prediction)
      self.R2[i]  = computations.metrics.R2( Y,prediction)
      # resampling
      jack = np.zeros(X.shape[0]-1)
      for b in range(self.B):
        # Jackknife sample
        X_jack = np.delete(X_train,b,0)
        Y_jack = np.delete(Y_train,b,0)
        # regression
        self.LinReg.fit(X_jack,Y_jack,alpha=lmbda)
        prediction = self.LinReg.predict(X_jack)
        jack[i]    = computations.metrics.MSE(Y_jack,prediction)
        # update command-line progress bar
        progress.update()
      # estimate bias and variance
      SUM  = np.mean(jack)
      self.Bias[i] = (self.B-1.)*(self.MSE[i]-SUM)/self.B
      self.Var[i]  = (self.B-1.)*np.sum((jack-SUM)**2)/self.B
  
