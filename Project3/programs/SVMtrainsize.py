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
behave with respect to the training set size. The SVMs are scored according
to their accuracy score.

Program parameters:
  C_lin           | boundary violation for SVM w/ linear kernel
  C_poly          | boundary violation for SVM w/ polynomial kernel
  C_rbf           | boundary violation for SVM w/ radial gaussian kernel
  C_sig           | boundary violation for SVM w/ sigmoid kernel
  train_size_min  | minimum training set size.
  train_size_max  | maximum training set size.
  trian_size_N    | number of training set sizes to consider.
  tag             | extra save tag used in np.save/np.load of results
  production_path | path for produced results

parameters used in tag = 1
-------------------
C_lin          = 1e-1
C_poly         = 1e-1
C_rbf          = 1e0
C_sig          = 1e-3
train_size_min = 0.1
train_size_max = 0.9
train_size_N   = 9

parameters used in tag = 2
-------------------
C_lin          = 1e-1
C_poly         = 1e-1
C_rbf          = 1e0
C_sig          = 1e-3
train_size_min = 0.1
train_size_max = 0.9
train_size_N   = 18

parameters used in tag = 3
-------------------
C_lin          = 1e-1
C_poly         = 1e-1
C_rbf          = 1e0
C_sig          = 1e-3
train_size_min = 0.05
train_size_max = 0.95
train_size_N   = 30
"""

# master parameters
load_production = True
tag             = sys.argv[1]

# parameters for production
C_lin          = 1e-1
C_poly         = 1e-1
C_rbf          = 1e0
C_sig          = 1e-3
train_size_min = 0.05
train_size_max = 0.95
train_size_N   = 30

# load pulsar data | IPP = Integrated Pulse Profile, DM-SNR = DispersionMeasure - Signal-Noise-Ratio,
path       =  "../data/pulsar_stars.csv"
Predictors = np.loadtxt(path,dtype=np.float_,skiprows=1,delimiter=",",usecols=(0,1,2,3,4,5,6,7))
Targets    = np.loadtxt(path,dtype=np.int_,  skiprows=1,delimiter=",",usecols=(8)).ravel()
Targets[Targets==0] = -1

# scale predictor space
scaler     = StandardScaler()
Predictors = scaler.fit_transform(Predictors)

# load previous produced results
if load_production:
  [[train_size],[Accuracy,Timing]] = np.load("../results/SVM/npy/violation{:s}.npy".format(tag))

# produce new results
else:
  # generate arrays
  train_size = np.linspace(train_size_min,train_size_max,train_size_N)
  Accuracy   = np.zeros((4,train_size_N))
  Timing     = np.zeros((4,train_size_N))
  
  # setup SVM
  Classifier = SVC(max_iter=-1,tol=1e-3)
  
  # production loops
  sanity = progress_bar(4*train_size_N)
  for i,k,c in zip(range(4),["linear","poly","rbf","sigmoid"],[C_lin,C_poly,C_rbf,C_sig]):
    # adjust SVM kernel
    Classifier.set_params(kernel=k,C=c)
    for j,t in enumerate(train_size):
      # divide data into training and test sets
      Predictors_train,Predictors_test,Targets_train,Targets_test = \
      train_test_split(Predictors,Targets,train_size=t,test_size=1-t)
      
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
  np.save("../results/SVM/npy/violation{:s}.npy".format(tag),[[train_size],[Accuracy,Timing]])

# plot results
fig,axes = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(5,9))

fig.suptitle("SVM train set size behaviour",x=0.56,y=0.97,fontsize=20)

axes[0].plot(100*train_size,Accuracy[0],label="linear")
axes[0].plot(100*train_size,Accuracy[1],label="polynomial")
axes[0].plot(100*train_size,Accuracy[2],label="radial")
axes[0].plot(100*train_size,Accuracy[3],label="sigmoid")

axes[1].plot(100*train_size,Timing[0],label="linear")
axes[1].plot(100*train_size,Timing[1],label="polynomial")
axes[1].plot(100*train_size,Timing[2],label="radial")
axes[1].plot(100*train_size,Timing[3],label="sigmoid")

axes[0].set_xlim([10,90])
axes[1].set_xlim([10,90])

axes[0].legend(loc='best')
axes[1].legend(loc='best')

axes[1].set_xlabel("training set size [%]",fontsize=18)

axes[0].set_ylabel("accuracy",            fontsize=18)
axes[1].set_ylabel("training times [sec]",fontsize=18)

fig.tight_layout()
fig.subplots_adjust(top=0.93)

fig.savefig("../results/SVM/img/trainsetsize{:s}.png".format(tag))

