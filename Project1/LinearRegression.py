import sys,os,shutil
import numpy as np
import numpy.linalg as la
import numpy.random as rand

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
    method     | regression method used
    resampling | bool that indicates whether to employ resampling techniques
  """
  # initial setup
  def __init__(self,X,Y,foldername=None):
    # setup folder system
    if not os.path.isdir("./results"): os.mkdir("./results")
    if foldername is None:
          self.dirpath = "./results/{:d}".format(len([sub[0] for sub in os.walk("./results")]))
    else: self.dirpath = "./results/"+str(foldername)
    if os.path.isdir(self.dirpath):
      duplicate     = sum([self.dirpath[10:] in subdir for subdir in [k[0][10:] for k in os.walk("./results/")]])
      self.dirpath += "{:d}".format(duplicate+1)
    os.mkdir(self.dirpath); self.dirpath += '/'
    # setup regression
    self.X_input    = X if len(X.shape) == 2 else X[:,None]
    self.Y_output   = Y if len(Y.shape) == 2 else Y[:,None]
    self.N          = int(len(Y))
    self.p          = int(len(X.T))
    self.model      = {}
    self.method     = "OLS"
    self.technique  = ""
    self.resampling = False
  
  # add a model to the analysis
  def add_model(self, F, name=None):
    if name is None: name = len(self.model)
    if name in self.model.keys():
      print("Model name '{:s}' is unavailable.".format(name))
      sys.exit(1)
    self.model[name] = RegressionModel(F,name)
  
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
      # regression without resampling
      if not self.resampling:
        self.model[key].regress(self.X_input, self.Y_output,self.method,alpha)
        self.model[key].compute_R2(self.X_input,self.Y_output)
      # regression with resampling
      if self.resampling:
        l, MSE_sample, R2_sample = 0, 0, 0
        if self.technique == "Bootstrap": X_test,Y_test,s = self.X_input,self.Y_output,np.size(self.Y_output)
        for k in range(self.K):
          if   self.technique == "Kfold":
            K       = k*self.K
            X_test  = self.X_input [K:K+self.K,:]
            Y_test  = self.Y_output[K:K+self.K,:]
            X_train = np.concatenate((self.X_input [:K,:],self.X_input [K+self.K:,:]))
            Y_train = np.concatenate((self.Y_output[:K,:],self.Y_output[K+self.K:,:]))
          elif self.technique == "Bootstrap":
            index   = rand.randint(s,size=s)
            X_train = self.X_input [index,:]
            Y_train = self.Y_output[index,:]
          self.model[key].regress(X_train,Y_train,self.method,alpha)
          self.model[key].compute_R2(X_test,Y_test)
          MSE_sample += self.model[key].MSE
          R2_sample  += self.model[key].R2
          l          += 1.
        self.model[key].MSE_sample, self.model[key].R2_sample = MSE_sample/l, R2_sample/l
      self.model[key].NumPy_save(self.dirpath,self.method,self.technique,alpha,self.resampling)

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
  def __init__(self, F, name):
    self.F    = F
    self.name = name
    self.run  = False
    self.MSE  = None
    self.R2   = None

  # perform regression
  def regress(self, X_data, Y_data, method, alpha):
    design_matrix = self.F(X_data)
    # Singular Value Decomposition
    U,d,VT = la.svd(design_matrix,full_matrices=False)
    V      = VT.T
    # compute regression coefficients
    if method in ["OLS","Ridge"]:
      sigma2 = np.sum(np.square(Y_data-self(X_data)))/(Y_data.shape[0]-X_data.shape[1])
      if method == "OLS":
        self.beta     = V@np.diag(1/d)@U.T@Y_data
        self.std_beta = np.sqrt(np.diag(V)*(1/d**2)*np.diag(VT)*sigma2)[:,None]
      if method == "Ridge":
        self.beta     = V@np.diag(d/(d**2+alpha))@U.T@Y_data
        self.std_beta = np.sqrt(np.diag(V)*np.power(1+alpha/d**2,-2)*np.diag(VT)*sigma2)[:,None]
    elif method == "Lasso": pass
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
  def NumPy_save(self, dirpath, method, technique, alpha,resampled=False):
    if self.run:
      if not resampled:
        path   = dirpath + "{:s}_{:s}".format(method,self.name)
        MSE,R2 = self.MSE,self.R2
      else:
        path   = dirpath + "{:s}_{:s}_{:s}".format(method,self.name,technique)
        MSE,R2 = self.MSE_sample,self.R2_sample
      np.save(path,[np.array([MSE,R2,alpha]),self.beta,self.std_beta])
  
if __name__ == "__main__":
  import matplotlib.pyplot as plt
  # real curve and polynomial approximation maps
  f  = lambda x: np.exp(-x**2)
  def f1(X):
    x = X.flatten()
    return np.array([np.ones(len(x)),np.exp(-x**2)]).T
  def f2(X):
    x = X.flatten()
    return np.array([np.ones(len(x)),x**2,x**4,x**6]).T
  # data set
  x        = np.linspace(-2,2,15)
  x_smooth = np.linspace(x.min(),x.max(),1000)
  y        = f(x) + 0.1*rand.randn(len(x))
  # initialise regression
  LinReg = LinearRegression(x,y)
  LinReg.add_model(F=f1,name="f1")
  LinReg.add_model(F=f2,name="f2")
  # perform analysis
  LinReg.use_method("Ridge")
  LinReg.run_analysis(alpha=1)
  F1,F2 = [LinReg.model[key] for key in ["f1","f2"]]
  LinReg.setup_resampling("Bootstrap",K=100)
  LinReg.run_analysis(alpha=1)
  # plot results
  plt.plot(x_smooth,f(x_smooth),label="underlying curve")
  plt.scatter(x,y,label="data points")
  plt.plot(x_smooth,F1(x_smooth),label="f1")
  plt.plot(x_smooth,F2(x_smooth),label="f2")
  plt.legend(loc="best")
  plt.show()
  # print results
  for m,F in zip(["f1","f2"],[F1,F2]):
    print("\n")
    print("model {:s}:".format(m))
    print("  sampled MSE = {:f}".format(LinReg.model[m].MSE_sample))
    print("  sampled R2  = {:f}".format(LinReg.model[m].R2_sample))
    print("  beta:")
    print(F.beta)
    print("  std(beta):")
    print(F.std_beta)
    print("  std(beta)/beta:")
    print(F.std_beta/F.beta)
  
  
  
