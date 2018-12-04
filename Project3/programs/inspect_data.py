import numpy as np
import matplotlib.pyplot as plt

# load pulsar data | IPP = Integrated Pulse Profile, SNR = DispersionMeasure - Signal-Noise-Ratio,
path         = "../data/pulsar_stars.csv"
IPP          = np.loadtxt(path,dtype=np.float_,skiprows=1,delimiter=",",usecols=(0,1,2,3)).T
SNR          = np.loadtxt(path,dtype=np.float_,skiprows=1,delimiter=",",usecols=(4,5,6,7)).T
Target_class = np.loadtxt(path,dtype=np.int_,  skiprows=1,delimiter=",",usecols=(8)).ravel()

# identify pulsars and non-pulsars
idx_pulsar      = np.argwhere(Target_class==1).ravel()
idx_nonpulsar   = np.argwhere(Target_class==0).ravel()
pulsar_range    = np.arange(1,len(idx_pulsar)+1)
nonpulsar_range = np.arange(1,len(idx_nonpulsar)+1)

# plot histograms
stat_titles    = ["mean value","standard deviation","kurtosis","skewness"]
savefig_labels = ["mean","std","kurt","skew"]

figs = [plt.figure()         for _   in range(8)]
axes = [fig.add_subplot(111) for fig in figs]
for i in range(4):
  # IPP histogram
  axes[i].hist(IPP[i][idx_nonpulsar],bins=150,label="non-pulsars")
  axes[i].hist(IPP[i][idx_pulsar],   bins= 50,label="pulsars")
  axes[i].set_title("Integrated Pulse Profile\nDistribution of {:s}".format(stat_titles[i]))
  axes[i].set_xlabel(stat_titles[i])
  axes[i].set_ylabel("frequency")
  axes[i].legend()
  figs[i].savefig("../results/inspection_figures/IPP_{:s}.png".format(savefig_labels[i]))
  
  # SNR histogram
  axes[i+4].hist(SNR[i][idx_nonpulsar],bins=150,label="non-pulsars")
  axes[i+4].hist(SNR[i][idx_pulsar],   bins= 50,label="pulsars")
  axes[i+4].set_title("Dispersion Measure - Signal-to-Noise Ratio\nDistribution of {:s}".format(stat_titles[i]))
  axes[i+4].set_xlabel(stat_titles[i])
  axes[i+4].set_ylabel("frequency")
  axes[i+4].legend()
  figs[i+4].savefig("../results/inspection_figures/SNR_{:s}.png".format(savefig_labels[i]))

plt.show()

