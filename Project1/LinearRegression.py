import sys,os,shutil
import numpy as np
import numpy.linalg as la
import numpy.random as rand
from sklearn.linear_model import lars_path

class LinearRegression():
  """
  Linear Regression Analysis
  
  The two fundamental components of this class is the output vetor Y, and the matrix of input
  vectors X = [X_1 --- X_p]. To investigate the relationship between X and Y, the user will
  impose a model (which itself may be a combination of models) using the map F = [f_1 --- f_p].
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
                   f_i(X) for i=1,...,l.
    
  Attributes:
    X_input    | input data (multivariate)
    Y_output   | output data (univariate)
    N          | number of data points
    p          | number of input vectors
    model      | dictionary containing all the user-defined models
    method     | current regression method
    save       | indicates whether to write results to file
    dirpath    | path to directory where data is stored in case save=True
  """
  # initial setup
  def __init__(self,X,Y,foldername=None,save=False):
    # setup folder system
    if save:
      if not os.path.isdir("./results"): os.mkdir("./results")
      if foldername is None:
            self.dirpath = "./results/{:d}".format(len([sub[0] for sub in os.walk("./results")]))
      else: self.dirpath = "./results/"+str(foldername)
      if os.path.isdir(self.dirpath):
        duplicate     = sum([self.dirpath[10:] in subdir for subdir in [k[0][10:] for k in os.walk("./results/")]])
        self.dirpath += "{:d}".format(duplicate+1)
      os.mkdir(self.dirpath); self.dirpath += '/'
    else: self.dirpath = "./"
    # setup regression
    self.X_input    = X if len(X.shape) == 2 else X[:,None]
    self.Y_output   = Y if len(Y.shape) == 2 else Y[:,None]
    self.N          = int(len(Y))
    self.p          = int(len(X.T))
    self.model      = {}
    self.method     = "OLS"
    self.save       = save
  
  # add a model to the analysis
  def add_model(self, F, name=None):
    if name is None: name = len(self.model)
    if name in self.model.keys():
      print("Model name '{:s}' is unavailable.".format(name))
      sys.exit(1)
    self.model[name] = RegressionModel(F,name)
  
  # remove a model from the analysis
  def remove_model(self,name):
    if name in self.model.keys(): del self.model[name]
  
  # select regression method
  def use_method(self, method):
    if method in ["OLS","Ridge","Lasso"]:
      self.method = method
    else: print("Invalid method, continuing with '{:s}'.".format(self.method))
  
  # perform regression analysis
  def run_analysis(self, alpha=0, technique="", K=1):
    for key in self.model.keys():
      if not(technique in ["Bootstrap","Kfold"]):
        self.model[key].regress(self.X_input, self.Y_output,self.method,alpha)
        self.model[key].compute_R2(self.X_input,self.Y_output)
      else:
        # preparation
        l, MSE_sample, R2_sample, Bias_sample, Var_sample = 0, 0, 0, 0, 0
        if technique == "Bootstrap": X_test,Y_test,s = self.X_input,self.Y_output,np.size(self.Y_output)
        # resampling algorithm
        for k in range(K):
          # divided data into Training and Testing sets
          if   technique == "Kfold":
            K_      = k*K
            X_test  = self.X_input [K_:K_+K,:]
            Y_test  = self.Y_output[K_:K_+K,:]
            X_train = np.concatenate((self.X_input [:K_,:],self.X_input [K_+K:,:]))
            Y_train = np.concatenate((self.Y_output[:K_,:],self.Y_output[K_+K:,:]))
          elif technique == "Bootstrap":
            index   = rand.randint(s,size=s)
            X_train = self.X_input [index,:]
            Y_train = self.Y_output[index,:]
          # perform regression analyis
          self.model[key].regress(X_train,Y_train,self.method,alpha)
          self.model[key].compute_R2(X_test,Y_test)
          MSE_sample  += self.model[key].MSE
          R2_sample   += self.model[key].R2
          Y_pred       = self.model[key](X_test)
          Bias_sample += np.mean((Y_test-np.mean(Y_pred))**2)
          Var_sample  += np.var(Y_pred)
          l           += 1.
        # Adjust final resampled quantities
        self.model[key].MSE_sample = MSE_sample/l
        self.model[key].R2_sample  = R2_sample/l
        self.model[key].Bias       = Bias_sample/l
        self.model[key].Var        = Var_sample/l
        # use the beta parameters from a non-resampled run
        self.save = False
        self.run_analysis(alpha)
        self.save = True
      # save results to NumPy file
      if self.save: self.model[key].NumPy_save(self.dirpath,self.method,alpha,technique,K)

# user-defined regression models
class RegressionModel():
  """
  Regression Map
  
  Support class for the LinearRegression class. Contains the model-specific data that
  is independent of other regression models.
  
  Constant attributes:
    F    | function map for the regression model, see LinearRegression() for details
    name | name of the model (used when writing to file)
  
  Variable attributes: (depends on the last usage of the attribute)
    run         | indicates whether a regression analysis as been performed
    beta        | column vector with model parameters
    std_beta    | column vector with standard deviation of model parameters
    MSE         | Mean Squared Error:                     E[(y-y_pred)^2]
    R2          | Coefficient of Determination:           1-MSE/E[(y-E[y])^2]
    MSE_sample  | Average Mean Squared Error:             E[MSE]
    R2_sample   | Average Coefficient of Determination:   E[R2]
    Bias_sample | Estimated Bias^2:                       E[(y-y_pred)]
    Var_sample  | Estimated Variance:                     Var(y_pred)
  """
  # sets up the model design matrix
  def __init__(self, F, name):
    self.F    = F
    self.name = name
    self.run  = False

  # perform regression
  def regress(self, X_data, Y_data, method, alpha):
    X = self.F(X_data)
    # Ordinary Least Squares and Ridge Regression
    if method in ["OLS","Ridge"]:
      U,d,VT = la.svd(X,full_matrices=False)
      V      = VT.T
      sigma2 = np.sum(np.square(Y_data-self(X_data)))/(Y_data.shape[0]-X_data.shape[1])
      if method == "OLS":
        self.beta     = V@np.diag(1/d)@U.T@Y_data
        self.std_beta = np.sqrt(np.diag(V)*(1/d**2)*np.diag(VT)*sigma2)[:,None]
      if method == "Ridge":
        self.beta     = V@np.diag(d/(d**2+alpha))@U.T@Y_data
        self.std_beta = np.sqrt(np.diag(V)*np.power(1+alpha/d**2,-2)*np.diag(VT)*sigma2)[:,None]
    # Lasso Regression
    elif method == "Lasso":
      alphas,_,Beta = lars_path(X,Y_data.flatten(),method="lasso")
      i             = np.where(alphas>alpha)[0][-1]
      self.beta     = Beta[:,i]
      self.std_beta = np.empty(self.beta.shape)
    self.run = True
    
  # compute model prediction
  def __call__(self, X_data):
    if self.run: return self.F(X_data)@self.beta
    else:        return 0
  
  # compute Mean Square Error
  def compute_MSE(self, X_data, Y_data):
    if self.run:
      self.MSE = np.mean(np.square(Y_data-self(X_data).flatten()))
      return self.MSE
    else: return 0
  
  # compute R2-score
  def compute_R2(self, X_data, Y_data):
    if self.run:
      y_flat = Y_data.flatten()
      self.compute_MSE(X_data,y_flat)
      if len(Y_data) == 1: self.MSE, self.R2 = 0, 1
      else:
        Y_bar   = np.mean(y_flat,dtype=np.float64)
        self.R2 = 1 - self.MSE/np.mean(np.square(y_flat-Y_bar))  # TODO : fix division by zero
      return self.R2
    else: return 0
  
  # save results to NumPy array
  def NumPy_save(self, dirpath, method, alpha, technique, K):
    if self.run:
      if technique == "":
        path   = dirpath + "{:s}_{:s}".format(str(method),str(self.name))
        np.save(path,[np.array([self.MSE,self.R2,alpha]),self.beta,self.std_beta])
      else:
        path = dirpath + "{:s}_{:s}_{:s}{:s}".format(str(method),str(self.name),str(technique),str(K))
        np.save(path,[np.array([self.MSE_sample,self.R2_sample,alpha,self.Bias,self.Var]),self.beta,self.std_beta])

