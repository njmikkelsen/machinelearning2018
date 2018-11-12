from IsingData import *
from LogisticClassification import *
from misclib import *
import numpy as np
import matplotlib.pyplot as plt

# verify folders
if not os.path.isdir("./output"):        os.mkdir("./output")
if not os.path.isdir("./output/LogReg"): os.mkdir("./output/LogReg")

# program parameters
name = "save"  # name of logistic classifier to load

# setup data
critical_data = IsingData.two_dim(data_config='critical')
critical_data.pad_ones()
target = critical_data.T.flatten()

# load classifier
LogReg = LogisticClassifier.load("./output/LogReg/{:s}".format(name))

# evaluate classifier with respect to 'critical' data set
prediction = LogReg.predict(critical_data.X).flatten()
Accuracy   = accuracy(target,prediction)
print("accuracy(critical) = {:f}".format(Accuracy))




