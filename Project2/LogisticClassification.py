import os
import numpy as np
from misclib import progress_bar
from sklearn.metrics import accuracy_score
try:
  import cPickle as pickle
except:
  import pickle

class LogisticClassifier(object):
  """
  Classification via Logistic Regression & Standard Gradient Descent
  --------------------------------------------------------------------
  """
  def __init__(self): 
    self.regressed = False  # indicates whether classifier is fitted
    self.penalized = False  # indicates whether loss function is penalized
  
  # save classifier
  def save(self,path_name):
    if self.regressed:
      path = path_name + ".pkl"
      if os.path.isfile(path): os.remove(path)
      with open(path,'wb') as Regressor:
        pickle.dump(self,Regressor)
    else:
      print('Oops! Classifier has not been trained!')
  
  # load classifier
  @classmethod
  def load(cls,path_name):
    path = path_name + ".pkl"
    with open(path,'rb') as Regressor:
      return pickle.load(Regressor)

  # add L2 penalty to logistic regression
  def add_penalty(self,lmbda):
    self.lmbda     = lmbda
    self.penalized = True
  
  # evaluate the logistic function
  def sigmoid(self,x):
    return 1./(1.+np.exp(-x))
    
  # evaluate the loss function
  def loss(self,T,Y):
    if self.penalized:
      return (-T*np.log(Y) - (1-T)*np.log(1-Y)).mean() + 0.5*self.lmbda*np.sum(self.beta.T@self.beta)
    else:
      return (-T*np.log(Y) - (1-T)*np.log(1-Y)).mean()
  
  # fit logreg to training set (X,Y) using stochastic gradient descent w/ batch training
  def fit(self,X,Y,epochs=10,N_batch=10,gamma=1e-2,alpha=0,track_accuracy=None):
    # prepare variables
    self.beta      = np.zeros((X.shape[1],1))
    batch_size     = int(X.shape[0]/N_batch)
    train_progress = progress_bar(epochs*N_batch)
    if not track_accuracy is None:
      self.accuracy_train = np.zeros(epochs)
      self.accuracy_test  = np.zeros(epochs)
    # training loop
    for epoch in range(epochs):
      for i in range(N_batch):
        # load random batch
        k = np.random.choice(N_batch)*batch_size
        if k < (N_batch-1)*batch_size:
          X_ = X[k:k+batch_size,:]
          T  = Y[k:k+batch_size,:]
        else:
          X_ = X[k:,:]
          T  = Y[k:,:]
        # update parameters
        h          = self.sigmoid(X_@self.beta)
        self.beta += gamma*(X_.T@(T-h))/T.size
        if self.penalized: self.beta += self.lmbda*self.beta
        train_progress.update()
      # update learning rate (exponential decay)
      gamma *= np.exp(-alpha)
      # track evolution of accuracy scores
      if not track_accuracy is None:
        h_train     = self.sigmoid(X@self.beta).flatten()
        h_test      = self.sigmoid(track_accuracy[0]@self.beta).flatten()
        pred_train  = (h_train > 0.5).astype(np.int8)
        pred_test   = (h_test  > 0.5).astype(np.int8)
        self.accuracy_train[epoch] = accuracy_score(Y.flatten(),      pred_train)
        self.accuracy_test[epoch]  = accuracy_score(track_accuracy[1],pred_test)
    self.regressed = True
  
  def predict(self,X):
    if self.regressed:
      Xbeta = X@self.beta
      h     = self.sigmoid(Xbeta)
      return (h > 0.5).astype(np.int8).flatten()
    else: return None
  
  
  
