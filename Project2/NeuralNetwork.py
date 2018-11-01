import sys,os
import numpy as np
from misclib import progress_bar
try:
  import cPickle as pickle
except:
  import pickle

# verify Neural Network directory exists
if not os.path.isdir("./neuralnetworks/"):
  os.mkdir("./neuralnetworks/")

class NeuralNetwork(object):
  """
  Parent class for Multi-Layered Perceptron Neural Network objects:
  MLPRegressor and MLPClassifier
  ---------------------------------------------------------------------
  The network is built on the back-propagation algorithm.
  The mathematics are constructed such that the input matrix X is
  expected to represent each sample with a column vector.
  ---------------------------------------------------------------------
  The output layer is automatically assigned 'identity' activation
  unless hidden=False. The default activation for a hidden layer is 'tanh'.
  """
  def __init__(self,p,N_samples,loss_function):
    self.p            = p          # the number of input predictors
    self.N_samples    = N_samples  # the number of samples
    self.N_layers     = 0          # the number of layers in network (incl. output)
    self.output_layer = False      # indicates whether the output layer has been explicitly defined
    self.network      = []         # network of layers
    self.layers_init  = False      # indicates whether layers have been given weights
    self.penalized    = False      # indicates whether loss function is given a penalty
    
    # setup the loss function for the neural network
    self.loss = NeuralNetworks.Loss(loss_function)
  
  # save/load network via pickle
  """
  usage:
  name    = "example"
  network = NeuralNetwork.load_network(name)
  network.save_network(name)
  """
  @classmethod
  def load_network(cls,network_path):
    with open("./network_path/{:s}.pkl".format(network_name),'rb') as Network:
      return pickle.load(Network)
  def save_network(self,network_name):
    with open("./network_path/{:s}.pkl".format(network_name),'wb') as Network:
      pickle.dump(self,Network)
  
  # add a layer to the network
  def add_layer(self,N_nodes,activation='tanh',idx=-1,hidden=True,std_weights=1,const_bias=0.01):
    # avoid placing layers behind the output layer
    if not(self.output_layer) and idx==-1:
      print('Error: the output layer has been defined, cannot place layer at the end of network.')
      return 0
    # determine layer input size
    if self.N_layers == 0: N_inputs = self.p                     # number of predictors/features
    else:                  N_inputs = self.network[idx].N_nodes  # number of nodes in previous layer
    # add layer to network
    self.network.insert(idx,NeuralNetwork.Layer(hidden, N_nodes,activation,hidden)
    self.output_layer = not hidden
    self.layers_init  = False
  
  # remove a layer from the network
  def remove_layer(self,idx):
    if idx in range(self.N_layers):
      del self.network[idx]
  
  # verify X-input dimensions
  def verify_X(self,X):
    return (X.shape[0]==self.p) and (X.shape[1]==self.N_samples)
  
  # verify Y-output dimensions
  def verify_Y(self,Y):
    return (X.shape[0]==self.network[-1].N_nodes) and (Y.shape[1]==self.N_samples)
  
  # initialize neural network with random weights and a constant biases
  def init_network(self):
    for i in range(self.N_layers):
      self.network[i].init_random_state()
    self.layers_init = True
  
  # run network | assumes self.layers_init == True to avoid an if test
  def feed_forward(self,X):
    Y = X
    for i in range(self.N_layers):
      Y = self.network[i].activate(Y)
    return Y

  # add a penalty to the loss function
  def add_penalty(self,lmbda):
    if lmbda > 0:
      self.lmbda     = lmbda
      self.penalized = True
    else:
      print("Error: invalid penalty lambda = '{:f}'. No penalty added".format(lmbda))

  # update the weights and biases of the neural network
  def update_network(self,grad_b,grad_W,eta):
    if self.penalized:
      for i in range(self.N_layers):
        grad_W[i] += lmbda*self.network[i].weight  # TODO : implement an arbitrary L^p penalty
    for i in range(self.N_layers):
      self.network[i].weight += -eta*grad_W[i]
      self.network[i].bias   += -eta*grad_b[i]
   
  def back_propagate_network(self,X,Y,T):
    """
    Back-propagation algorithm
    -----------------------------------------------------------------------------------
    Back-propagates the output layer error based on inputs X, outputs Y and targets T.
    Returns the gradients of the network's weights and biases.
    """
    # collect network activations and activation derivatives
    a,da = [X],[]
    for i in range(self.N_layers):
      a.append( self.network[i].activation.a)
      da.append(self.network[i].activation.df(a[i+1]))
    # back-propagate errors
    d = [da[-1]*self.loss.dL]
    for i in range(2,self.N_layers+1):
      d.insert(0,self.network[-i+1].weight.T@d[0] * da[-i])
    # evaluate gradients
    grad_b,grad_W = [],[]
    for i in range(self.N_layers):
      grad_b.append(np.sum(d[i],axis=1,keepdims=True)/Y.shape[1])
      grad_W_3d = np.einsum('ij,jk->jik',d[i],a[i].T)
      grad_W.append(np.sum(grad_W_3d,axis=0)/Y.shape[1])
    return grad_b,grad_W
  
  # train neural network
  def train(self,X,Y,iterations=100,epochs=1,N_batch=10,eta=5e-1,lmbda=1e-2):
    """
    Train network using a stochastic gradient descent algorithm w/ mini-batches
    ----------------------------------------------------------------------------
    """
    if not self.layers_init:
      print("Error: layers uninitialized"); return None
    if not isinstance(epochs,'int') and epochs>0:
      print("Error: invalid epochs"); return None
    if not self.verify_X(X) or not self.verify_Y(Y):
      print("Error: data shapes do not align with neural networks"); return None
    batch_size     = int(self.X_data.shape[1]/N_batch)
    train_progress = progress_bar(epochs*iterations)  # command-line progress bar
    for epoch in range(epochs):
      for i in range(iterations):
        # load random batch
        k = np.random.choice(N_batch)*batch_size
        if k < (N_batch-1)*batch_size:
          X = self.X_data[:,k:k+batch_size]
          T = self.Y_data[:,k:k+batch_size]
        else:
          X = self.X_data[:,k:]
          T = self.Y_data[:,k:]
        # feed forward and back propagation
        Y             = self.feed_forward(X)
        grad_b,grad_W = self.back_propagate_network(X,Y,T)
        # update network weights and biases
        self.update_network(grad_b,grad_W,eta)                 # TODO : implement varying eta
        # update command-line progress bar
        train_progress.update()
  
  class Layer(object):
    """
    Neural Network Layer
    ---------------------
    The weight matrix for this layer is of shape: N_nodes x N_inputs
    The bias vector for this layer is of length:  N_nodes
    """
    def __init__(self,N_nodes,N_inputs,activation,hidden,std_weight,const_bias):
      self.hidden     = hidden      # True = hidden layer, False = output layer
      self.N_nodes    = N_nodes     # number of nodes in layer
      self.N_inputs   = N_inputs    # number of inputs to layer
      self.std_weight = std_weight  # initial standard deviation of random weights
      self.const_bias = const_bias  # initial bias
      
      # prepare layer activation function
      self.activation = NeuralNetwork.Activation(activation)
    
    # initialize normally distributed weights & a constant biases
    def init_random_state(self):
      self.weight = self.std_weight * np.random.randn(self.N_nodes,self.N_inputs)
      self.bias   = self.init_bias  + np.zeros((self.N_nodes,1))
    
    # evaluate activation function
    def activate(self,X):
      return self.activation.f(self.weight@X + self.bias)
  
  class Loss(object):
    """
    Loss function for neural network
    ---------------------------------------
    available loss functions:
      'squared_error', 'cross_entropy'
    """
    def __init__(self,loss_function):
      try:
        exec("""
        self.L  = self.{:s}
        self.dL = self.{:s}_derivative
        """.format(loss_function,loss_function))
      except:
        print("Warning: loss function '{:s}' not found, using 'squared_error' instead".format(loss_function))
    
    # squared error
    def squared_error(self,T,Y):
      return 0.5*np.sum(T-Y)
    def squared_error_derivative(self,T,Y):
      return T-Y
    
    # cross_entropy
    def cross_entropy(self,T,Y):
      return -np.sum(T*np.log(Y))
    def cross_entropy(self,T,Y):
      return (Y-T)/(Y*(1-Y))
    
  class Activation(object):
    """
    Activation function for neural network layer.
    ----------------------------------------------------------------------------
    Each network layer is attributed an accompanying 'Activation' instance.
    The class tracks the layer's previous activation and activation derivative.
    
    The derivatives are written in terms of their respective activations due to
    numerical considerations. Ex: d/dx(tanh) = sech^2 x = 1 - tanh^2 x
    
    available activation functions include:
      'logit', 'tanh', 'softmax', 'identity'
    """
    def __init__(self,activation):
      try:
        exec("""
        self.f  = self.{:s}
        self.df = self.{:s}_derrivative
        """.format(activation,activation))
      except:
        print("Warning: activation function '{:s}' not found, using 'tanh' instead".format(activation))
        self.f  = self.tanh
        self.df = self.tanh_derivative
    
    # logit activation
    def logit(self,x):
      return 1./(1.+np.exp(-x))
    def logit_derivative(self,a):
      return a*(1-a)
    
    # tanh activation
    def tanh(self,x):
      return np.tanh(x)
    def tanh_derivative(self,a)
      return 1-a**2
    
    # softmax activation
    def softmax(self,x):
      exp = np.exp(x-np.max(x))
      return exp/np.sum(exp,axis=0)
    def softmax_derivative(self,a):
      s = a.reshape(-1,1)
      return np.diagflat(s) - (s)@(s.T)
    
    # identity activation
    def identity(self,x):
      return x
    def identity_derivative(self,a):
      return 1.
    

class MLPRegressor(NeuralNetwork):
  """
  Regression via the Multi-Layered Perceptron Neural Network
  """
  def __init__(self,p,N_samples,loss_function="squared_error"):
    super().__init__(p,N_samples,loss_function)

  def predict(self,X):
    if self.layers_init:
      if self.verify_X(X)
        return self.feed_forward(X)
    else: return None

class MLPClassifier(NeuralNetwork):
  """
  Classification via the Multi-Layered Perceptron Neural Network
  """
  def __init__(self,p,N_samples,loss_function="cross_entropy"):
    super().__init__(p,N_samples,loss_function)

  def predict(self,X):
    if self.layers_init:
      if self.verify_X(X)
        return np.max(self.feed_forward(X),axis=0)
    else: return None

