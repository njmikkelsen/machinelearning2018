import sys,os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from progress_bar import progress_bar

"""
Program for analysing how radial Gaussian support vector machines and the pulsar data set
behave with respect to the hyperparameter of the radial Gassuian kernel. The SVMs are
score according to the their accuracy score.

Program parameters:
  C               | boundary violation for SVM
  train_size      | portion of full data set used for training
  gamma_min       | minimum gamma
  gamma_max       | maximum gamma
  gamma_N         | number of gamma values to iterate over (logarithmically spaced)
  tag             | extra save tag used in np.save/np.load of results
  production_path | path for produced results

parameters used in tag = 1
-------------------
train_size = 0.20
C          = 1e0
gamma_min  = 1e-6
gamma_max  = 1e+2
gamma_N    = 200
"""

# master parameters
load_production = False
tag             = sys.argv[1]

# parameters for production
train_size = 0.20
C          = 1e0
gamma_min  = 1e-6
gamma_max  = 1e+2
gamma_N    = 200

# load pulsar data | IPP = Integrated Pulse Profile, DM-SNR = DispersionMeasure - Signal-Noise-Ratio,
path       =  "../data/pulsar_stars.csv"
Predictors = np.loadtxt(path,dtype=np.float_,skiprows=1,delimiter=",",usecols=(0,1,2,3,4,5,6,7))
Targets    = np.loadtxt(path,dtype=np.int_,  skiprows=1,delimiter=",",usecols=(8)).ravel()
Targets[Targets==0] = -1

# scale predictor space
scaler     = StandardScaler()
Predictors = scaler.fit_transform(Predictors)

# divide data into training and test sets
Predictors_train,Predictors_test,Targets_train,Targets_test = \
train_test_split(Predictors,Targets,train_size=train_size,test_size=1-train_size)

# load previous produced results
if load_production:
  [gamma,Accuracy,Timing] = np.load("../results/SVM/npy/radial{:s}.npy".format(tag))

# produce new results
else:
  # generate arrays
  gamma      = np.logspace(np.log10(gamma_min),np.log10(gamma_max),gamma_N)
  Accuracy   = np.zeros(gamma_N)
  Timing     = np.zeros(gamma_N)
  
  # setup SVM
  Classifier = SVC(kernel="rbf",C=C,max_iter=-1,tol=1e-3)
  
  # production loops
  sanity = progress_bar(gamma_N)
  for i,g in enumerate(gamma):
    # adjust SVM kernel shift
    Classifier.set_params(gamma=g)
    
    # train SVM
    time_0 = time.perf_counter()
    Classifier.fit(Predictors_train,Targets_train)
    time_1 = time.perf_counter()
    
    # score SVM
    Accuracy[i] = Classifier.score(Predictors_test,Targets_test)
    Timing[i]   = time_1-time_0
    
    # update command line progress bar
    sanity.update()
  # save new results
  np.save("../results/SVM/npy/radial{:s}.npy".format(tag),[gamma,Accuracy,Timing])

# plot results
fig,axes = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(5,9))

fig.suptitle("radial Gaussian SVM behaviour",x=0.56,y=0.97,fontsize=20)

axes[0].plot(gamma,Accuracy)
axes[1].plot(gamma,Timing)

axes[0].set_xscale("log")
axes[1].set_xscale("log")
#axes[1].set_yscale("log")

axes[1].set_xlabel(r"scaling factor $\gamma$",fontsize=18)
axes[0].set_ylabel("accuracy",                fontsize=18)
axes[1].set_ylabel("training times [sec]",    fontsize=18)

fig.tight_layout()
fig.subplots_adjust(top=0.93)

fig.savefig("../results/SVM/img/radial{:s}.png".format(tag))
plt.show()
