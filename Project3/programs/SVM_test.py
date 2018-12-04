import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# load pulsar data | IPP = Integrated Pulse Profile, SNR = DispersionMeasure - Signal-Noise-Ratio,
path       =  "../data/pulsar_stars.csv"
Predictors = np.loadtxt(path,dtype=np.float_,skiprows=1,delimiter=",",usecols=(0,1,2,3,4,5,6,7))
Targets    = np.loadtxt(path,dtype=np.int_,  skiprows=1,delimiter=",",usecols=(8)).ravel()
Targets[Targets==0] = -1

# divide data into training and test sets
train_size = 0.95
Predictors_train,Predictors_test,Targets_train,Target_test = \
train_test_split(Predictors,Targets,train_size=train_size,test_size=1-train_size)

# initialize and train Neural Network
Classifier = SVC(gamma="auto")
Classifier.fit(Predictors_train,Targets_train)

# score results
accuracy = Classifier.score(Predictors_test,Target_test)
print("Accuracy = {:f}".format(accuracy))


