import sys
import numpy as np
import numpy.linalg as la

class LinearRegression():
  """
  Linear Regression Analysis
  
  The two fundamental components of this class is the output vetor Y, and the matrix of input
  vectors X = [X_1 --- X_p]. To investigate the relationship between X and Y, the user will
  imposes a model (which itself may be a combination of models) using the map F = [f_1 --- f_p].
  The user is free to impose as many models as required; in particular, the different models may
  be compared with respect to their MSE scores, R2-scores, confidence intervals, etc. Note that
  each individual component f_i(X) does not necessarily need to use all components of X.

  The fundamental assumption of this class is that Y may be written as a linear combination of
  the mapped vectors F(X) in the following manner:
  
                                             l
    Y = f_1(X)B_1 + ... + f_l(X)B_l + E  =  SUM f_i(X)B_i + E  =  F(X)B + E
                                            i=1
  
  where B_1,...,B_l are the model's linear coefficients and E is a vector containing the model's
  normally distributed random errors.
  
  Practical issue: In order for the user-defined F(X) map to work with LinearRegression(), it
                   must a) accept a single matrix argument X whose columns represent each input
                   vector, and b) return a design matrix whose columns represent each mapping
                   f_i(X) i=1,...,l.
    
  Attributes:
    X_input    | input data (multivariate)
    Y_output   | output data (univariate)
    N          | number of data points
    p          | number of input vectors
    model      | dictionary containing all the user-defined models
    method     | regression method used
    resampling | bool that indicates whether to employ resampling techniques
  """
  # initial setup
  def __init__(self,X,Y):
    self.X_input    = X if len(X.shape) == 2 else X[:,None]
    self.Y_output   = Y if len(Y.shape) == 2 else Y[:,None]
    self.N          = int(len(Y))
    self.p          = int(len(X.T))
    self.model      = {}
    self.method     = "OLS"
    self.resampling = False
  
  # add a model to the analysis
  def add_model(self, F, name=None):
    if name is None: name = len(self.model)
    if name in self.model.keys():
      print("Model name '{:s}' is unavailable.".format(name))
      sys.exit(1)
    self.model[name] = RegressionModel(F)
  
  # select regression method
  def use_method(self, method):
    if method in ["OLS","Ridge","Lasso"]:
      self.method = method
    else: print("Invalid method, continuing with '{:s}'.".format(self.method))
  
  # setup data resampling
  def setup_resampling(self, technique,K=1):
    if technique in ["Kfold","Bootstrap"]:
      self.resampling = True
      self.technique  = technique
      self.K          = K  # Kfold: #samples. Bootstrap: #iterations
  
  # perform regression analysis
  def run_analysis(self, alpha=0):
    for key in self.model.keys():
      # regression with resampling
      if self.resampling:
        if self.technique == "Kfold":
          l         = 0
          MSE_cross = 0
          R2_cross  = 0
          for k in range(0,len(self.Y_output),self.K):
            X_test  = self.X_input [k:k+self.K,:]
            Y_test  = self.Y_output[k:k+self.K,:]
            X_train = np.concatenate((self.X_input [:k,:],self.X_input [k+self.K:,:]))
            Y_train = np.concatenate((self.Y_output[:k,:],self.Y_output[k+self.K:,:]))
            self.model[key].regress(X_train,Y_train,self.method,alpha)
            self.model[key].compute_R2(X_test,Y_test)
            MSE_cross += self.model[key].MSE
            R2_cross  += self.model[key].R2
            l         += 1.
          MSE_cross, R2_cross = MSE_cross/l, R2_cross/l
        elif self.technique == "Bootstrap":
          pass
      # regression without resampling
      else:
        self.model[key].regress(self.X_input, self.Y_output,self.method,alpha)
        self.model[key].compute_R2(self.X_input,self.Y_output)
        

# user-defined regression models
class RegressionModel():
  """
  Regression Map
  
  Support class for the LinearRegression class. Contains the model-specific data that
  is independent of other models.
  
  Attributes:
    F    | model map (defines the model)
    X    | model design matrix
    beta | model coefficients with respect to some data set Y_data
    run  | bool that indicates whether the model coefficients have been determined
    MSE  | Mean Squared Error with respect to some data set Y_data
    R2   | R2-score (coefficient of determination) with respect to some data set Y_data
  """
  # sets up the model design matrix
  def __init__(self, F):
    self.F   = F
    self.run = False

  # perform regression
  def regress(self, X_data, Y_data, method, alpha):
    # Singular Value Decomposition
    design_matrix = self.F(X_data)
    U,d,VT        = la.svd(design_matrix,full_matrices=False)
    # compute regression coefficients
    if   method == "OLS":   self.beta = VT.T@np.diag(1/d)@U.T@Y_data
    elif method == "Ridge": self.beta = VT.T@np.diag(d/(d**2+alpha))@U.T@Y_data
    elif method == "Lasso": pass
    self.run = True
  
  # compute model prediction
  def __call__(self, X_data):
    if self.run: return self.F(X_data)@self.beta
    else:        return 0
  
  # compute Mean Square Error
  def compute_MSE(self, X_data, Y_data):
    if self.run:
      self.MSE = np.mean(np.power(Y_data-self(X_data).flatten(),2))
      return self.MSE
    else: return 0
  
  # compute R2-score
  def compute_R2(self, X_data, Y_data):
    if self.run:
      y_flat = Y_data.flatten()
      self.compute_MSE(X_data,y_flat)
      Y_bar   = np.mean(y_flat,dtype=np.float64)
      self.R2 = 1 - self.MSE/np.mean(np.power(y_flat-Y_bar,2))  # TODO : fix division by zero
      return self.R2
    else: return 0
  
  




if __name__ == "__main__":
  import numpy.random as rand
  import matplotlib.pyplot as plt
  # real curve and polynomial approximation maps
  f  = lambda x: x + np.cos(x)
  def f1(X):
    x = X.flatten()
    return np.array([np.ones(len(x)),x,x**2,x**3]).T
  def f2(X):
    x = X.flatten()
    return np.array([np.ones(len(x)),x,x**2,x**3,x**4,x**5]).T
  # data set
  x        = np.linspace(-5,5,15)
  x_smooth = np.linspace(x.min(),x.max(),1000)
  y        = f(x) + 0.5*rand.randn(len(x))
  # initialise regression
  LinReg = LinearRegression(x,y)
  LinReg.add_model(F=f1)
  LinReg.add_model(F=f2)
  # perform analysis
  LinReg.use_method("Ridge")
  LinReg.setup_resampling("Kfold",K=2)
  LinReg.run_analysis(alpha=1)
  # plot results
  plt.plot(x_smooth,f(x_smooth))
  for key in LinReg.model.keys():
    A = LinReg.model[key]
    plt.plot(x_smooth,A(x_smooth))
  plt.show()
  
      
    
  
  
  
  
  
  
  
  
