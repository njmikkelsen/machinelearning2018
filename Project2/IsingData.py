import sys,os
import numpy as np
from sklearn.model_selection import train_test_split
try:
  import cPickle as pickle
except:
  import pickle

# verify data directories exist
if not os.path.isdir('./IsingData'):
  os.mkdir('./IsingData')
  os.mkdir('./IsingData/one_dim')
  os.mkdir('./IsingData/two_dim')
if not os.path.isdir('./IsingData/one_dim'):
  os.mkdir('./IsingData/one_dim')
if not os.path.isdir('./IsingData/two_dim'):
  os.mkdir('./IsingData/two_dim')


class IsingData(object):
  """
  Data-loading object for ease of use in FYS-STK 3155 - Project 2.
  All loaded data is automatically stored under './IsingData/one_dim' or './IsingData/two_dim'
  depending on whether the data is one-dimensional or two-dimensional.
  
  The Ising-model is a binary model that attempts to describe the macroscopic properties of a
  large system whose microscopic components are arragned in an n-dimensional lattice. Note that
  this class only considers one-dimensional and two-dimensional lattices.
  
  In order to avoid potential boundary problems, the model assumes periodic boundary
  conditions. Originally used to describe ferromagnetism as a result of neighbouring spin-spin
  interactions, the model assigns each component a spin of +1 or -1 and assumes all interactions
  share a common coupling/interaction strength J.
  The resulting Hamiltonian (energy) of the system is thus:
  
    H = - J * SUM  s1*s2
            <s1,s2>
              
  where <s1,s2> indicates a summation over all neighbouring lattice-components.
  """
  
  class one_dim(object):
    """
    One-dimensional Ising-data. Note that the one-dimensional Hamiltonian simplifies to
                N
        H = -J SUM s_j s_(j+1)
               j=1
    where s_(j+1) = s_1.
    -------------------------------------------------------------------------------
    Data specifications:
      N | dimensionality of the spin-configuration (number of spins in the system)
      J | coupling/interaction strength
      M | number of data samples
    The X data array is a matrix whose columns represent each spin-configuration.
    The T data array is a 1-dimensional array.
    -------------------------------------------------------------------------------
    The additional parameter 'new_data' provides the ability to generate a new data
    set regardless whether an earlier data set already exists.
    Default behaviour is to load the previously generated data set if it exists.
    """
    # initialize data set
    def __init__(self,N,J,M,sigma_s=0.5,sigma_H=2,new_data=False):
      self.N,self.J,self.M,self.sigma_s,self.sigma_H = N,J,M,sigma_s,sigma_H
      self.datadir    = "./IsingData/one_dim/{:d}_{:f}_{:d}_{:f}_{:f}/".format(N,J,M,sigma_s,sigma_H)
      self.transposed = False  # indicates whether data is transposed
      self.data_split = False  # indicates whether data is split into training and test sets
      self.padded     = False  # indicates whether data has been padded with ones (for intercept)
      self.check_exists()
      if not self.exists or new_data:
        self.generate_data()
        if not self.exists:
          np.save(self.datadir+"X.npy",self.X)
          np.save(self.datadir+"T.npy",self.T)
      else:
        self.X = np.load(self.datadir+"X.npy")
        self.T = np.load(self.datadir+"T.npy")
    
    # check if data has already been generated
    def check_exists(self):
      self.exists = True if os.path.isdir(self.datadir) else False
      if not self.exists:
        os.mkdir(self.datadir)
    
    # generate a new data set
    def generate_data(self):
      """
      This function is based on code from this Notebook:
      https://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB_CVI-linreg_ising.html
      """
      # init spin-configurations
      states = np.random.choice([-1,1],size=(self.M,self.N))
      states = np.asarray(states,dtype=np.int8)
      # find non-zero contributions to Hamiltonian
      E = np.zeros((self.N,self.N),)
      for i in range(self.N):
        E[i,(i+1)%self.N] -= self.J
      # compute Hamiltonian
      H = np.einsum('...i,ij,...j->...',states,E,states)
      # recast states array for regression
      states = np.einsum('...i,...j->...ij',states,states)
      states = states.reshape((states.shape[0],states.shape[1]*states.shape[2]))
      # store final data set (X,T) = (inputs + error, targets + error)
      self.X = states + self.sigma_s*np.random.randn(*states.shape)
      self.T = H      + self.sigma_H*np.random.randn(*H.shape)

    # pad a column of ones to X matrix (such that the intercept can be modelled)
    def pad_ones(self):
      # pad full data
      if self.transposed: self.X = np.c_[np.ones(self.X.shape[1]),self.X.T].T
      else:               self.X = np.c_[np.ones(self.X.shape[0]),self.X]
      # pad split data
      if self.data_split:
        if self.transposed:
          self.X_train = np.c_[np.ones(self.X_train.shape[1]),self.X_train.T].T
          self.X_test  = np.c_[np.ones(self.X_test.shape[1]), self.X_test.T].T
        else:
          self.X_train = np.c_[np.ones(self.X_train.shape[0]),self.X_train]
          self.X_test  = np.c_[np.ones(self.X_test.shape[0]), self.X_test]
      self.padded = True
    
    # transpose the input data
    def transpose(self):
      # transpose full data
      self.X = self.X.T
      # transpose split data
      if self.data_split:
        self.X_train = self.X_train.T
        self.X_test  = self.X_test.T
      self.tranposed = not self.transposed
    
    # divide data set into training and test sets
    def split(self,ratio=0.5):
      """
      0 <= ratio <= 1  | (size of training set) / (size of test set)
      """
      if 0 <= ratio <= 1:
        if self.transposed:
          self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X.T,self.T,test_size=1-ratio)
        else:
          self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X,self.T,test_size=1-ratio)
        self.data_split = True
      else:
        print("Error: ratio out of bounds: 0<=ratio<=1")
    
    # convert a single-dimensional J parameter vector to a two-dimensional J parameter matrix
    @staticmethod
    def J_vec_to_matrix(J):
      N = int(np.sqrt(np.size(J)))
      return np.array(J).reshape((N,N))
      

  class two_dim(object):
    """
    Two-dimensional Ising-data that was originally downloaded from:
    https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/
    -----------------------------------------------------------------------------------------------
    The two-dimensional data has been organized such that different data sets may be loaded
    according to the parameter 'data_config'.
    
    The available data configurations are:
      all      | this includes every data set
      order    | this includes all data sets with temperatures: 0.25 <= T <=  1.75 (70 000 samples)
      critical | this includes all data sets with temperatures: 2.00 <= T <=  2.50 (30 000 samples)
      disorder | this includes all data sets with temperatures: 2.75 <= T <=  4.00 (60 000 samples)
      noncrit  | this includes configurations 'order' and 'disorder'
      random   | this includes a random number of samples from any data set, uniformly distributed
    """
    def __init__(self,data_config='all'):
      self.transposed = False
      self.data_split = False
      self.NumPyfy_data()
      if   data_config in ['all','order','critical','disorder','noncrit']:
        self.load(data_config)
      elif data_config == 'random':
        self.load_random()
      else:
        print('Error: invalid data configuration.')
        print('Using all data sets instead')
        self.load('all')
      self.X = self.X.astype(np.int8)
      self.T = self.T[:,None]
    
    # load the desired data sets
    def load(self,data_config):
      self.X = np.load('./IsingData/two_dim/NumPyfied/X_{:s}.npy'.format(data_config))
      self.T = np.load('./IsingData/two_dim/NumPyfied/Y_{:s}.npy'.format(data_config))
    
    # load a random number of samples from any data sets
    def load_random(self):
      X         = np.load('./IsingData/two_dim/NumPyfied/X_all.npy')
      T         = np.load('./IsingData/two_dim/NumPyfied/Y_all.npy')
      N_samples = np.random.randint(1,X.shape[1])
      idx       = np.random.choice(np.arange(X.shape[1]),size=N_samples)
      self.X    = X[:,idx]
      self.T    = T[idx]
    
    # divide data set into training and test sets
    def split(self,ratio=0.5):
      """
      0 <= ratio <= 1  | (size of training set) / (size of test set)
      """
      if 0 <= ratio <= 1:
        if self.transposed:
          self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X.T,self.T,test_size=1-ratio)
          self.X_train,self.X_test = self.X_train.T,self.X_test.T
        else:
          self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X,self.T,test_size=1-ratio)
        self.data_split = True
      else:
        print("Error: ratio out of bounds: 0<=ratio<=1")
    
    # pad a column of ones to X matrix (such that the intercept can be modelled)
    def pad_ones(self):
      if self.transposed:
        self.X = np.c_[np.ones(self.X.shape[1],dtype=np.int8),self.X.T].T
      else:
        self.X = np.c_[np.ones(self.X.shape[0],dtype=np.int8),self.X]
    
    # transpose the input data
    def transpose(self):
      # transpose full data
      self.X = self.X.T
      # transpose split data
      if self.data_split:
        self.X_train = self.X_train.T
        self.X_test  = self.X_test.T
      self.transposed = not self.transposed
    
    # transform 1-dim target vector to a one-hot vector for use in neural networks
    def make_hot(self):
      self.T = to_categorical_numpy(self.T.flatten()).T
      if self.data_split:
        self.Y_train = to_categorical_numpy(self.Y_train.flatten()).T
        self.Y_test  = to_categorical_numpy(self.Y_test.flatten()).T
    
    def NumPyfy_data(self):
      """
      This function loads all the data from the original data files 'Ising2DFM_reSample_L40_T=All.pkl'
      and 'Ising2DFM_reSample_L40_T=All_labels.pkl', and saves it as NumPy files.
      -----------------------------------------------------------------------------------------------
      This function is based on code from this Notebook:
      https://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB_CVII-logreg_ising.html
      """
      path = './IsingData/two_dim/NumPyfied/'
      if not os.path.isdir(path):
        # make NumPyfied directory
        os.mkdir(path)
        # load spin-configurations
        states = pickle.load(open("./IsingData/two_dim/Ising2DFM_reSample_L40_T=All.pkl",'rb'))
        states = np.unpackbits(states).astype(np.int8).reshape(-1, 1600)
        states[np.where(states==0)]=-1
        # load phase data
        phases = pickle.load(open("./IsingData/two_dim/Ising2DFM_reSample_L40_T=All_labels.pkl",'rb'))
        # reorganize data
        X_order    = states[:70000,:]
        X_critical = states[70000:100000,:]
        X_disorder = states[100000:,:]
        
        Y_order    = phases[:70000]
        Y_critical = phases[70000:100000]
        Y_disorder = phases[100000:]
        
        X_noncrit = np.concatenate([X_order,X_disorder])
        Y_noncrit = np.concatenate([Y_order,Y_disorder])
        # save data in NumPy files
        np.save(path+'X_all.npy',     states)        
        np.save(path+'X_order.npy',   X_order)
        np.save(path+'X_critical.npy',X_critical)
        np.save(path+'X_disorder.npy',X_disorder)
        np.save(path+'X_noncrit.npy', X_noncrit)

        np.save(path+'Y_all.npy',     phases)
        np.save(path+'Y_order.npy',   Y_order)
        np.save(path+'Y_critical.npy',Y_critical)
        np.save(path+'Y_disorder.npy',Y_disorder)
        np.save(path+'Y_noncrit.npy', Y_noncrit)

# one-hot in numpy
def to_categorical_numpy(integer_vector):
  """
  This function was copied from the lecture notes on Neural Networks found here:
  https://compphysics.github.io/MachineLearning/doc/pub/NeuralNet/html/._NeuralNet-bs046.html
  """
  n_inputs = len(integer_vector)
  n_categories = np.max(integer_vector) + 1
  onehot_vector = np.zeros((n_inputs, n_categories))
  onehot_vector[range(n_inputs), integer_vector] = 1
  return onehot_vector
  
if __name__ == '__main__':
  # generates/prepares the standard data sets if not already done
  one_dim = IsingData.one_dim(N=40,J=1,M=10000,new_data=False)
  two_dim = IsingData.two_dim()
    
