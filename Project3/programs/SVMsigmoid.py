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
Program for analysing how sigmoid support vector machines and the pulsar data set
behave with respect to the hyperparameter of the sigmoid kernel. The SVMs are
score according to the their accuracy score.

Program parameters:
  C               | boundary violation for SVM
  train_size      | portion of full data set used for training
  shift_min       | minimum shift
  shift_max       | maximum shift
  shift_N         | number of shift values to iterate over (logarithmically spaced)
  tag             | extra save tag used in np.save/np.load of results
  production_path | path for produced results

parameters used in tag = 1
-------------------
train_size = 0.20
C          = 1e-3
shift_min  = 1e-6
shift_max  = 1e+6
shift_N    = 50

parameters used in tag = 2
-------------------
train_size = 0.20
C          = 1e-3
shift_min  = 1e-12
shift_max  = 1e+0
shift_N    = 50

parameters used in tag = 3
-------------------
train_size = 0.20
C          = 1e-3
shift_min  = 1e+0
shift_max  = 1e+12
shift_N    = 50
"""

# master parameters
load_production = True
tag             = sys.argv[1]

# parameters for production
train_size = 0.20
C          = 1e-3
shift_min  = 1e-6
shift_max  = 1e+6 
shift_N    = 50

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
  [shift,Accuracy,Timing] = np.load("../results/SVM/npy/sigmoid{:s}.npy".format(tag))

# produce new results
else:
  # generate arrays
  shift      = np.logspace(np.log10(shift_min),np.log10(shift_max),shift_N)
  Accuracy   = np.zeros(shift_N)
  Timing     = np.zeros(shift_N)
  
  # setup SVM
  Classifier = SVC(kernel="sigmoid",C=C,max_iter=-1,tol=1e-3)
  
  # production loops
  sanity = progress_bar(shift_N)
  for i,r in enumerate(shift):
    # adjust SVM kernel shift
    Classifier.set_params(coef0=r)
    
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
  np.save("../results/SVM/npy/sigmoid{:s}.npy".format(tag),[shift,Accuracy,Timing])

# plot results
fig,axes = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(5,9))

fig.suptitle("sigmoid SVM behaviour",x=0.56,y=0.97,fontsize=20)

axes[0].plot(shift,Accuracy)
axes[1].plot(shift,Timing)

axes[0].set_xscale("log")
axes[1].set_xscale("log")
#axes[1].set_yscale("log")

axes[1].set_xlabel(r"argument shift",     fontsize=18)
axes[0].set_ylabel("accuracy",            fontsize=18)
axes[1].set_ylabel("training times [sec]",fontsize=18)

fig.tight_layout()
fig.subplots_adjust(top=0.93)

fig.savefig("../results/SVM/img/sigmoid{:s}.png".format(tag))
plt.show()
