import sys,os
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from progress_bar import progress_bar

"""
Program for analysing how neural networks and the pulsar data set behave
with respect to hyperparameter:
  - train_size | train-to-test ratio in division of data into training and test sets
  
NOTE: This program is designed to produce results that are visualized using the program 'NNgridsearch3_visualize.py'.
  
On the structure of the neural networks:
  All hidden layers have the same number of nodes.
  All hidden layers have the same activation function.

On the performance metrics:
  The performance of the networks will be judged according to 3 metrics:
    1) final accuracy score
    2) required number of training epochs
    3) training time

On the program's structure:
  The program consists primarily of a for-loop that iterates over the
  neural network's hyperparameter.
  The data produced here is visualized in "NNgridsearch3_visualize.py".

Program parameters:
  L               | number of layers
  N               | number of nodes per layer
  gamma0          | initial learning rate
  eta             | strength of learning momentum
  train_size_min  | minimum train_size
  train_size_max  | maximum train_size
  train_size_N    | number of train_size values to iterate over
  overwrite       | whether to overwrite existing results: True or False
  tag             | extra save tag used in np.save/np.load of results

parameters used in tag = 1
-------------------
gamma0 = 0.1
eta    = 0.3

parameters used in tag = 2
-------------------
gamma0 = 0.01
eta    = 0.5

parameters used in tag = 3
-------------------
gamma0 = 0.001
eta    = 0.7
"""
# master parameters
overwrite = True
tag       = sys.argv[1]

# parameters for production
L              = int(sys.argv[2])
N              = int(sys.argv[3])
gamma0         = float(sys.argv[4])
eta            = float(sys.argv[5])
train_size_min = 0.1
train_size_max = 0.9
train_size_N   = 100

# generate train_size array
train_size = np.linspace(train_size_min,train_size_max,train_size_N)

# load pulsar data | IPP = Integrated Pulse Profile, DM-SNR = DispersionMeasure - Signal-Noise-Ratio,
path       =  "../data/pulsar_stars.csv"
Predictors = np.loadtxt(path,dtype=np.float_,skiprows=1,delimiter=",",usecols=(0,1,2,3,4,5,6,7))
Targets    = np.loadtxt(path,dtype=np.int_,  skiprows=1,delimiter=",",usecols=(8)).ravel()
Targets[Targets==0] = -1


# load previous produced results
if os.path.exists("../results/NN/npy/gridsearch3_{:d}_{:d}_{:f}_{:f}_{:s}.npy".format(L,N,gamma0,eta,tag)) and not overwrite:
  print("Results file already exists and overwrite = False")
  sys.exit(1)

# prepare result arrays
Accuracy = np.zeros((train_size_N,3))
Epochs   = np.zeros((train_size_N,3))
Timing   = np.zeros((train_size_N,3))

# production loops
sanity = progress_bar(3*train_size_N)
for j,activation in enumerate(["logistic","tanh","relu"]):
  # setup network
  Classifier = MLPClassifier( \
    hidden_layer_sizes = [N for _ in range(L)],
    activation         = activation,
    solver             = "sgd",
    learning_rate_init = gamma0,
    momentum           = eta,
    learning_rate      = "adaptive",
    max_iter           = 100000)
  for i,t in enumerate(train_size):
    # divide data into training and test sets
    Predictors_train,Predictors_test,Targets_train,Targets_test = \
    train_test_split(Predictors,Targets,train_size=t,test_size=1-t)
    
    # train network
    time_0 = time.perf_counter()
    Classifier.fit(Predictors_train,Targets_train)
    time_1 = time.perf_counter()
    
    # score network
    accuracy = Classifier.score(Predictors_test,Targets_test)
    
    # store results
    Accuracy[i,j] = accuracy
    Epochs[i,j]   = Classifier.n_iter_
    Timing[i,j]   = time_1-time_0
    
    # update command line progress bar
    sanity.update()
# save new results
np.save("../results/NN/npy/gridsearch3_{:d}_{:d}_{:s}.npy".format(L,N,tag), \
[[train_size],[Accuracy,Epochs,Timing]])

