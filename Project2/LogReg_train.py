from IsingData import *
from LogisticClassification import *
from misclib import *
from plotlib import LogRegPlot_AccuracyEvolution
import numpy as np
import matplotlib.pyplot as plt

# verify folders
if not os.path.isdir("./output"):        os.mkdir("./output")
if not os.path.isdir("./output/LogReg"): os.mkdir("./output/LogReg")

# parameters
r        = 0.3      # training portion of sampless
epochs   = 40       # number of epochs training cycles
N_batch  = 10       # number of batches/subdivisions of X_train & Y_train
gamma    = 5e-3     # step length in gradient descent
alpha    = 1e-1     # step length fall off
lmbda    = 1e-3     # L2 penalty
name     = "save"   # savefig name
save     = True     # whether to save results

# setup data
data = IsingData.two_dim(data_config='noncrit')
data.pad_ones()
data.split(r)

# Logistic regression
LogReg = LogisticClassifier()
LogReg.add_penalty(lmbda)
LogReg.fit(data.X_train,data.Y_train,epochs,N_batch,gamma,alpha,track_accuracy=[data.X_test,data.Y_test])
predict_train = LogReg.predict(data.X_train)
predict_test  = LogReg.predict(data.X_test)

# plot evolution
print("final test     accuracy = {:f}".format(LogReg.accuracy_test[-1]))
print("final training accuracy = {:f}".format(LogReg.accuracy_train[-1]))
fig = LogRegPlot_AccuracyEvolution(LogReg.accuracy_test,LogReg.accuracy_train)
fig.savefig("./output/LogReg/AccuracyEvolution_{:s}.png".format(name))
plt.show()

# save regression parameters
if save:
  LogReg.save("./output/LogReg/{:s}".format(name))


"""
the standard:

r        = 0.3      # training portion of sampless
epochs   = 40       # number of epochs training cycles
N_batch  = 10       # number of batches/subdivisions of X_train & Y_train
gamma    = 5e-3     # step length in gradient descent
alpha    = 1e-1     # step length fall off
lmbda    = 1e-3     # L2 penalty


the long run:

r        = 0.3      # training portion of sampless
epochs   = 200       # number of epochs training cycles
N_batch  = 10       # number of batches/subdivisions of X_train & Y_train
gamma    = 5e-4     # step length in gradient descent
alpha    = 1e-3     # step length fall off
lmbda    = 1e-3     # L2 penalty
name     = "the_long_run"   # savefig name
"""
