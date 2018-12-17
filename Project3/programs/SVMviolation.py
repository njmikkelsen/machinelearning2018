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
Program for analysing how support vector machines and the pulsar data set
behave with respect to SVM violation strength. The SVMs are scored according
to their accuracy score.

Program parameters:
  C_min           | minimum violation
  C_max           | maximum violation
  C_N             | number of violation values to iterate over (note that violation values are logarithmically spaced)
  tag             | extra save tag used in np.save/np.load of results
  production_path | path for produced results

parameters used in tag = 1
-------------------
C_min      = 1e-4
C_max      = 1e1
C_N        = 15

parameters used in tag = 2
-------------------
C_min      = 1e-4
C_max      = 1e1
C_N        = 50

parameters used in tag = 3
-------------------
C_min      = 1e-5
C_max      = 1e2
C_N        = 100
"""

# master parameters
load_production = True
tag             = sys.argv[1]

# parameters for production
C_min      = 1e-5
C_max      = 1e2
C_N        = 100
train_size = 0.5

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
  [[C],[Accuracy,Timing]] = np.load("../results/SVM/npy/violation{:s}.npy".format(tag))

# produce new results
else:
  # generate arrays
  C        = np.logspace(np.log10(C_min),np.log10(C_max),C_N)
  Accuracy = np.zeros((4,C_N))
  Timing   = np.zeros((4,C_N))
  
  # setup SVM
  Classifier = SVC(max_iter=-1,tol=1e-3)
  
  # production loops
  sanity = progress_bar(4*C_N)
  for i,k in enumerate(["linear","poly","rbf","sigmoid"]):
    # adjust SVM kernel
    Classifier.set_params(kernel=k)
    for j,c in enumerate(C):
      # adjust SVM violation
      Classifier.set_params(C=c)
      
      # train SVM
      time_0 = time.perf_counter()
      Classifier.fit(Predictors_train,Targets_train)
      time_1 = time.perf_counter()
      
      # score SVM
      Accuracy[i,j] = Classifier.score(Predictors_test,Targets_test)
      Timing[i,j]   = time_1-time_0
      
      # update command line progress bar
      sanity.update()
  # save new results
  np.save("../results/SVM/npy/violation{:s}.npy".format(tag),[[C],[Accuracy,Timing]])

# plot results
matplotlib.rcParams.update({'font.size' : 11})

fig,axes = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(5,9))

fig.suptitle("SVM violation behaviour",x=0.56,y=0.97,fontsize=20)

axes[0].plot(C,Accuracy[0],label="linear")
axes[0].plot(C,Accuracy[1],label="polynomial")
axes[0].plot(C,Accuracy[2],label="radial")
axes[0].plot(C,Accuracy[3],label="sigmoid")

axes[1].plot(C,Timing[0],label="linear")
axes[1].plot(C,Timing[1],label="polynomial")
axes[1].plot(C,Timing[2],label="radial")
axes[1].plot(C,Timing[3],label="sigmoid")

axes[0].set_xscale('log')
axes[1].set_xscale('log')

axes[0].legend(loc=3)
axes[1].legend(loc=2)

axes[1].set_xlabel("boundary violation strength",fontsize=18)

axes[0].set_ylabel("accuracy",            fontsize=18)
axes[1].set_ylabel("training times [sec]",fontsize=18)

fig.tight_layout()
fig.subplots_adjust(top=0.93)

fig.savefig("../results/SVM/img/violation{:s}.png".format(tag))
plt.show()

