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
  - L | number of layers          
  - N | number of nodes per layer
  
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
  L_min           | minimum L
  L_max           | maximum L
  L_N             | number of L values to iterate over
  N_min           | minimum N
  N_max           | maximum N
  N_N             | number of N values to iterate over
  train_size      | train-to-test ratio in division of data into training and test sets
  load_production | load produced results (instead of producing new results): True or False
  tag             | extra save tag used in np.save/np.load of results
  production_path | path for produced results

parameters used in tag = 1
-------------------
L_min      = 1
L_max      = 50
L_N        = 30
N_min      = 2
N_max      = 100
N_N        = 30

parameters used in tag = 2
-------------------
L_min      = 1
L_max      = 25
L_N        = 13
N_min      = 10
N_max      = 30
N_N        = 11

parameters used in tag = 3
-------------------
L_min      = 1
L_max      = 3
L_N        = 3
N_min      = 1
N_max      = 30
N_N        = 30
"""

# master parameters
load_production = True
tag             = sys.argv[2]

# parameters for production
#activation = "logistic"
activation = sys.argv[1]    # read activation from the command line
L_min      = 1
L_max      = 3
L_N        = 3
N_min      = 1
N_max      = 30
N_N        = 30
train_size = 0.5

# generate L and N arrays
L = np.linspace(L_min,L_max,L_N).astype(int)
N = np.linspace(N_min,N_max,N_N).astype(int)

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
  [[L],[N],[Accuracy,Epochs,Timing]] = np.load("../results/NN/npy/gridsearch1_{:s}{:s}.npy".format(activation,tag))

# produce new results
else:
  # prepare result arrays
  Accuracy = np.zeros((L_N,N_N))
  Epochs   = np.zeros((L_N,N_N))
  Timing   = np.zeros((L_N,N_N))

  # production loops
  sanity = progress_bar(L_N*N_N)
  for i,l in enumerate(L):
    for j,n in enumerate(N):
      # setup network
      Classifier = MLPClassifier( \
        hidden_layer_sizes = [n for _ in range(l)],
        activation         = activation,
        solver             = "sgd",
        max_iter           = 1000)
      
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
  np.save("../results/NN/npy/gridsearch1_{:s}{:s}.npy".format(activation,tag),[[L],[N],[Accuracy,Epochs,Timing]])

###
### Section 2: Visualization of results
###

# redefine default matplotlib rcParams
matplotlib.rcParams.update({'font.size'             : 14,
                            'figure.subplot.top'    : 0.9,
                            'figure.subplot.left'   : 0.20,
                            'figure.subplot.right'  : 0.86,
                            'figure.subplot.bottom' : 0.15,
                            'savefig.dpi'           : 300,
                            'image.cmap'            : "jet"   })

# prepare for plots
len_L = len(L)
len_N = len(N)
xlen  = np.min([5,len_N]).astype(int)
ylen  = np.min([5,len_L]).astype(int)

xticks = np.linspace(0,len_N-1,xlen)
yticks = np.linspace(0,len_L-1,ylen)

xticklabels = ["{:d}".format(n) for n in np.linspace(N.min(),N.max(),xlen).astype(int)]
yticklabels = ["{:d}".format(l) for l in np.linspace(L.min(),L.max(),ylen).astype(int)]

# additional thresholds for the images
if   tag == "1":
  epochs_max   = 100.
  timing_max   = 200.
elif tag == "2":
  epochs_max   = 100.  
  timing_max   = 40.
elif tag == "3":
  epochs_max   = 200.
  timing_max   = 15.

Epochs[Epochs > epochs_max] = None
Timing[Timing > timing_max] = None

# plot results as images
fig,axes = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(4.3,9.5))

im1 = axes[0].imshow(Accuracy,aspect=len_N/len_L)
im2 = axes[1].imshow(Epochs,  aspect=len_N/len_L)
im3 = axes[2].imshow(Timing,  aspect=len_N/len_L)

axes[2].set_xlabel("no. nodes per layer",fontsize=18)
for ax in axes:
  ax.set_ylabel("no. layers",fontsize=18)

axes[2].set_xticks(xticks)
axes[2].set_xticklabels(xticklabels)
for ax in axes:
  ax.set_yticks(yticks)
  ax.set_yticklabels(yticklabels)

fig.colorbar(im1,ax=axes[0],fraction=0.06)
fig.colorbar(im2,ax=axes[1],fraction=0.06)
fig.colorbar(im3,ax=axes[2],fraction=0.06)

fig.suptitle("{:s} activation".format(activation),x=0.5,y=0.99,fontsize=20)

axes[0].set_title("accuracy scores",     fontsize=18)
axes[1].set_title("number of epochs",    fontsize=18)
axes[2].set_title("training times [sec]",fontsize=18)

plt.tight_layout()
fig.subplots_adjust(top=0.91,right=0.83,bottom=0.07)

fig.savefig("../results/NN/img/gridsearch1_{:s}{:s}.png".format(activation,tag))


