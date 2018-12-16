import sys,os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from progress_bar import progress_bar

"""
Program for analysing how neural networks and the pulsar data set behave
with respect to hyperparameters:
  - gamma0 | initial learning rate
  - eta    | strength of learning momentum
  
On the structure of the neural networks:
  All hidden layers have the same number of nodes.
  All hidden layers have the same activation function.

On the performance metrics:
  The performance of the networks will be judged according to 3 metrics:
    1) final accuracy score
    2) required number of training epochs
    3) training time

On the program's structure:
  The program consists of 2 sections: production of results & visualization of results.
  The production section consists primarily of a double for-loop that iterates over the
  neural network's hyperparameters. In case results from previous runs already exist,
  then these results may be loaded using the load_production argument.
  The visualization section both creates and saves the program's output plots.

Program parameters:
  activation      | activation function used in hidden layers: "logistic", "tanh" or "relu".
  L               | number of layers
  N               | number of nodes per layer
  gamma0_min      | minimum gamma0
  gamma0_min      | maximum gamma0
  gamma0_N        | number of gamma0 values to iterate over (note that gamma0 values are logarithmically spaced)
  eta_min         | minimum eta
  eta_max         | maximum eta
  eta_N           | number of eta values to iterate over
  train_size      | train-to-test ratio in division of data into training and test sets
  load_production | load produced results (instead of producing new results): True or False
  tag             | extra save tag used in np.save/np.load of results
  production_path | path for produced results

parameters used in tag = 1
-------------------
L = 1
N = 8

parameters used in tag = 2
-------------------
L = 2
N = 15

parameters used in tag = 3
-------------------
L = 3
N = 30
"""

# master parameters
load_production = True
tag             = sys.argv[2]

# parameters for production
#activation = "logistic"
activation = sys.argv[1]
L          = 3
N          = 30
gamma0_min = 1e-4
gamma0_max = 1e+0
gamma0_N   = 17
eta_min    = 0.01
eta_max    = 0.99
eta_N      = 15
train_size = 0.5

# generate gamma0 and eta arrays
gamma0 = np.logspace(np.log10(gamma0_min),np.log10(gamma0_max),gamma0_N)
eta    = np.linspace(eta_min,eta_max,eta_N)

# load pulsar data | IPP = Integrated Pulse Profile, DM-SNR = DispersionMeasure - Signal-Noise-Ratio,
path       =  "../data/pulsar_stars.csv"
Predictors = np.loadtxt(path,dtype=np.float_,skiprows=1,delimiter=",",usecols=(0,1,2,3,4,5,6,7))
Targets    = np.loadtxt(path,dtype=np.int_,  skiprows=1,delimiter=",",usecols=(8)).ravel()
Targets[Targets==0] = -1

# divide data into training and test sets
Predictors_train,Predictors_test,Targets_train,Targets_test = \
train_test_split(Predictors,Targets,train_size=train_size,test_size=1-train_size)

###
### Section 1: Production of results
###

# load previous produced results
if load_production:
  [[gamma0],[eta],[Accuracy,Epochs,Timing]] = np.load("../results/NN/npy/gridsearch2_{:s}{:s}.npy".format(activation,tag))

# produce new results
else:
  # prepare result arrays
  Accuracy = np.zeros((gamma0_N,eta_N))
  Epochs   = np.zeros((gamma0_N,eta_N))
  Timing   = np.zeros((gamma0_N,eta_N))
  
  # setup network
  Classifier = MLPClassifier( \
    hidden_layer_sizes = [N for _ in range(L)],
    activation         = activation,
    solver             = "sgd",
    learning_rate      = "adaptive",
    max_iter           = 10000)

  # production loops
  sanity = progress_bar(gamma0_N*eta_N)
  for i,g in enumerate(gamma0):
    for j,n in enumerate(eta):
      # adjust parameters
      Classifier.set_params(learning_rate_init=g, momentum=n)
      
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
  np.save("../results/NN/npy/gridsearch2_{:s}{:s}.npy".format(activation,tag),[[gamma0],[eta],[Accuracy,Epochs,Timing]])

###
### Section 2: Visualization of results
###

# redefine default matplotlib rcParams
matplotlib.rcParams.update({\
'font.size'                   : 14,
'axes.formatter.use_mathtext' : True,
'figure.subplot.top'          : 0.9,
'figure.subplot.left'         : 0.20,
'figure.subplot.right'        : 0.86,
'figure.subplot.bottom'       : 0.15,
'savefig.dpi'                 : 300,
'image.cmap'                  : "jet" \
})

# prepare for plots
len_gamma0 = len(gamma0)
len_eta    = len(eta)
xlen       = np.min([5,len_eta   ]).astype(int)
ylen       = np.min([5,len_gamma0]).astype(int)

xticks = np.linspace(0,len_eta-1,   xlen)
yticks = np.linspace(0,len_gamma0-1,ylen)

xticklabels = ["{:.1f}".format(n) for n in np.linspace(eta.min(),eta.max(),xlen)]
yticklabels = ["{:.1f}".format(g) for g in np.linspace(np.log10(gamma0.min()),np.log10(gamma0.max()),ylen)]

# additional thresholds for the images
if   tag == "1":
  epochs_max   = 1000.
  timing_max   = 25.
elif tag == "2":
  epochs_max   = 1000.  
  timing_max   = 50.
elif tag == "3":
  epochs_max   = 1000.
  timing_max   = 40.

Epochs[Epochs > epochs_max] = None
Timing[Timing > timing_max] = None

# plot
fig,axes = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(4.3,9.5))

im1 = axes[0].imshow(Accuracy,aspect=len_eta/len_gamma0)
im2 = axes[1].imshow(Epochs,  aspect=len_eta/len_gamma0)
im3 = axes[2].imshow(Timing,  aspect=len_eta/len_gamma0)

axes[2].set_xlabel(r"$\eta$",fontsize=18)
for ax in axes:
  ax.set_ylabel(r"$\log(\gamma_0)$",fontsize=18)

axes[2].set_xticks(xticks)
axes[2].set_xticklabels(xticklabels)
for ax in axes:
  ax.set_yticks(yticks)
  ax.set_yticklabels(yticklabels)

fig.colorbar(im1,ax=axes[0],fraction=0.06)
fig.colorbar(im2,ax=axes[1],fraction=0.06)
fig.colorbar(im3,ax=axes[2],fraction=0.06)

fig.suptitle("{:s} activation".format(activation),x=0.5,y=0.99,fontsize=18)

axes[0].set_title("accuracy scores",     fontsize=18)
axes[1].set_title("number of epochs",    fontsize=18)
axes[2].set_title("training times [sec]",fontsize=18)

plt.tight_layout()
fig.subplots_adjust(top=0.92,right=0.83,bottom=0.07)

plt.savefig("../results/NN/img/gridsearch2_{:s}{:s}.png".format(activation,tag))
#plt.show()



