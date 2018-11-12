import os
import numpy as np
import matplotlib.pyplot as plt
from misclib import *

"""
THIS LIBRARY IS A WRAPPER FOR PLOTS IN PROJECT 2
-----------------------------------------------------------------------
The existence of this library is so that long and tedius plotting commands
may be removed from the main body of other programs used in the project.
"""

if not os.path.isdir("./output/"):
  os.mkdir("./output/")

def LinRegPlot_MSE_R2(Lmbda,OLS,Ridge,Lasso):
  """
  Function for plotting the MSE and R^2 scores of OLS, Ridge & Lasso regression
  as a function of regularization lambda.
  
  fig1,ax1 | MSE plot
  fig2,ax2 | R^2 plot
  """
  fig1,fig2 = plt.figure(),plt.figure()
  ax1,ax2   = fig1.add_subplot(111),fig2.add_subplot(111)
  
  # plot OLS data
  Lmin,Lmax = np.min(Lmbda),np.max(Lmbda)

  ax1.hlines(OLS.MSE_train,Lmin,Lmax,'b',linestyle='--')
  ax1.hlines(OLS.MSE_test, Lmin,Lmax,'b',linestyle='-',label="OLS")
  
  ax2.hlines(OLS.R2_train,Lmin,Lmax,'b',linestyle='--')
  ax2.hlines(OLS.R2_test, Lmin,Lmax,'b',linestyle='-',label="OLS")
  
  # plot Ridge data
  ax1.plot(Lmbda,Ridge.MSE_train,'g--')
  ax1.plot(Lmbda,Ridge.MSE_test, 'g-',label="Ridge")
  
  ax2.plot(Lmbda,Ridge.R2_train,'g--')
  ax2.plot(Lmbda,Ridge.R2_test, 'g-',label="Ridge")
  
  # plot Lasso data
  ax1.plot(Lmbda,Lasso.MSE_train,'r--')
  ax1.plot(Lmbda,Lasso.MSE_test, 'r-',label="Lasso")
  
  ax2.plot(Lmbda,Lasso.R2_train,'r--')
  ax2.plot(Lmbda,Lasso.R2_test, 'r-',label="Lasso")
  
  # make MSE plot pretty
  ax1.set_title("MSE Scores for Linear Regression Analysis of 1-dim Ising Data")
  ax1.set_xlabel(r"regularization $\lambda$")
  ax1.set_ylabel(r"MSE$\,(\lambda)$")
  ax1.set_xscale('log')
  ax1.legend(loc=2)
  ax1.grid(True)

  # make R2 plot pretty
  ax2.set_title(r"$R^2$ Scores for Linear Regression Analysis of 1-dim Ising Data")
  ax2.set_xlabel(r"regularization $\lambda$")
  ax2.set_ylabel(r"$R^2\,(\lambda)$")
  ax2.set_xscale('log')
  ax2.legend(loc=3)
  ax2.grid(True)
  
  return (fig1,ax1),(fig2,ax2)
  
def LinRegPlot_BiasVarDecomposition(LinReg,Lmbda,loglog=False):
  """
  Function for plotting the Bias-Variance decomposition of Ridge or Lasso regression
  as a function of the regularization parameter lambda.
  """
  fig = plt.figure()
  ax  = fig.add_subplot(111)

  # plot
  ax.plot(Lmbda,LinReg.MSE_test,  'b-',label='MSE')
  ax.plot(Lmbda,LinReg.Bias2_test,'g-',label=r'Bias$\,^2$')
  ax.plot(Lmbda,LinReg.Var_test,  'r-',label='Var')
  
  ax.plot(Lmbda,LinReg.MSE_train,  'b--')
  ax.plot(Lmbda,LinReg.Bias2_train,'g--')
  ax.plot(Lmbda,LinReg.Var_train,  'r--')
  
  # make plot pretty    
  ax.set_title("Bias-Variance decomposition of {:s} model".format(LinReg.regressor_type))
  ax.set_xlabel(r"regularization $\lambda$")
  ax.set_ylabel(r"MSE / Bias$\,^2$ / Var")
  ax.set_xscale('log')
  if loglog: ax.set_yscale('log')
  ax.legend(loc=2)
  ax.grid(True)
  
  return fig,ax

def LinRegPlot_Jmatrix(J,i,regressor,cmap_args,lmbda,lmbda_acc):
  """
  Function for plotting the J-matrix from an OLS, Ridge or Lasso regression analysis of one-dimensional data.
  i = 0,1,2  refers to OLS,Ridge,Lasso
  """
  fig = plt.figure()
  ax  = fig.add_subplot(111)
  im = ax.imshow(J,**cmap_args)

  # make plot pretty
  title_extra = "" if i==0 else r" with $\lambda={:s}\cdot10^{{{:s}}}$".format(*scientific_number(lmbda,lmbda_acc))
  ax.set_title(r"Matrix plot of $J$ coefficient from" + "\n{:s} Regression{:s}".format(regressor,title_extra))
  ax.set_xlabel(r"column index $k$")
  ax.set_ylabel(r"row index $j$")
  cbar = fig.colorbar(im)

  return fig

def NNRegPlot_CostEvolution(Regressor,loglog):
  """
  Function for plotting the evolution of the cost function of neural network regressor.
  """
  fig = plt.figure()
  ax  = fig.add_subplot(111)
  
  # plot
  ax.plot(Regressor.cost_train,label="training")
  ax.plot(Regressor.cost_test, label="test")
  ax.set_title("Evolution of the cost function")
  if loglog: ax.set_yscale("log")
  ax.set_xlabel("epoch")
  ax.set_ylabel("Cost")
  ax.legend()
  
  return fig

def NNRegPlot_Hyper(hyper,R2_train,R2_test,Hyper1,Hyper2,wide=False,log=True,lower_boundary=0):
  """
  Function for plotting the R^2 scores from neural network regressor as a function hyperparameters.
  
    hyper = 0 : Hyper1 = Lmbda,  Hyper2 = Eta
    hyper = 1 : Hyper1 = Epochs, Hyper2 = Alpha
  """
  
  # corresponds to hyper1
  if hyper==0:
    xlabel   = r"regularization log$\,_{10}(\lambda)$"
    ylabel   = r"learning rate log$\,_{10}(\eta)$"
  # corresponds to hyper2
  elif hyper==1:
    xlabel   = "number of epochs"
    ylabel   = r"learning rate falloff $\alpha$"
  
  fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,6.5))
  
  # plot 
  cmap_args = dict(vmin=lower_boundary,vmax=1.,cmap="gnuplot2")
  im1       = ax1.imshow(R2_train,**cmap_args)
  im2       = ax2.imshow(R2_test, **cmap_args)
  
  fig.subplots_adjust(bottom=0.2,left=0.13)
  cbar_ax = fig.add_axes([0.13,0.1, 0.76, 0.04])
  fig.colorbar(im2,cax=cbar_ax,orientation="horizontal")
  
  # make plot pretty
  fig.suptitle(r"$R^2$ scores of neural network",y=0.9)
  ax1.set_title('training')
  ax1.set_xlabel(xlabel)
  ax1.set_ylabel(ylabel)
  
  ax2.set_title('testing')
  ax2.set_xlabel(xlabel)  
  
  if wide:
    ax1.set_xticks(np.arange(len(Hyper1)))
    ax1.set_yticks(np.arange(len(Hyper2)))
    ax2.set_xticks(np.arange(len(Hyper1)))
    if log:
      ax1.set_xticklabels(["{:.1f}".format(k) for k in np.log10(Hyper1)])
      ax1.set_yticklabels(["{:.1f}".format(k) for k in np.log10(Hyper2)])    
      ax2.set_xticklabels(["{:.1f}".format(k) for k in np.log10(Hyper1)])
    else:
      ax1.set_xticklabels(["{:.1f}".format(k) for k in Hyper1])
      ax1.set_yticklabels(["{:.1f}".format(k) for k in Hyper2])    
      ax2.set_xticklabels(["{:.1f}".format(k) for k in Hyper1])
  else:
    ax1.set_xticks(np.arange(len(Hyper1))[::2])
    ax1.set_yticks(np.arange(len(Hyper2))[::2])
    ax2.set_xticks(np.arange(len(Hyper1))[::2])
    if log:
      ax1.set_xticklabels(["{:.1f}".format(k) for k in np.log10(Hyper1)[::2]])
      ax1.set_yticklabels(["{:.1f}".format(k) for k in np.log10(Hyper2)[::2]])    
      ax2.set_xticklabels(["{:.1f}".format(k) for k in np.log10(Hyper1)[::2]])
    else:
      ax1.set_xticklabels(["{:.1f}".format(k) for k in Hyper1[::2]])
      ax1.set_yticklabels(["{:.1f}".format(k) for k in Hyper2[::2]])
      ax2.set_xticklabels(["{:.1f}".format(k) for k in Hyper1[::2]])

  return fig

def LogRegPlot_AccuracyEvolution(accuracy_test,accuracy_train):
  """
  Program for plotting the evolution of the acccuracy test and train scores of a logistic regression
  optimisation via stochastic gradient descent with batch training.
  """
  fig = plt.figure()
  ax  = fig.add_subplot(111)
  
  # plot
  ax.plot(np.arange(len(accuracy_test)), accuracy_test, marker='.',label="test")
  ax.plot(np.arange(len(accuracy_train)),accuracy_train,marker='.',label="training")
  
  # make plot pretty
  ax.set_title("Evolution of accuracy scores of logistic regression")
  ax.set_xlabel("epochs")
  ax.set_ylabel("accuracy")
  
  return fig










