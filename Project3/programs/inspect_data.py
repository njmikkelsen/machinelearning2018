import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

# redefine default matplotlib rcParams
matplotlib.rcParams.update({'font.size'             : 10,
                            'figure.subplot.left'   : 0.20,
                            'figure.subplot.right'  : 0.86,
                            'figure.subplot.bottom' : 0.15,
                            'savefig.dpi'           : 300   })

# plot individual histograms
stat_titles    = ["mean value","standard deviation","kurtosis","skewness"]
savefig_labels = ["mean","std","kurt","skew"]

figs = [plt.figure(figsize=(3.4,3)) for _   in range(8)]
axes = [fig.add_subplot(111)        for fig in figs]
for i in range(4):
  # IPP histogram
  axes[i].hist(IPP[i][idx_nonpulsar],bins=150,label="non-pulsars",density=True)
  axes[i].hist(IPP[i][idx_pulsar],   bins= 50,label="pulsars",    density=True)
  axes[i].set_title("Integrated Pulse Profile\nNormalised distribution of {:s}".format(stat_titles[i]),fontsize=10)
  axes[i].set_xlabel(stat_titles[i],fontsize=10)
  axes[i].set_ylabel("probability",fontsize=10)
  axes[i].legend()
  figs[i].savefig("../results/inspection_figures/IPP_{:s}.png".format(savefig_labels[i]))
  
  # SNR histogram
  axes[i+4].hist(SNR[i][idx_nonpulsar],bins=150,label="non-pulsars",density=True)
  axes[i+4].hist(SNR[i][idx_pulsar],   bins= 50,label="pulsars",    density=True)
  axes[i+4].set_title("Dispersion Measure - Signal-to-Noise Ratio\nNormalised distribution of {:s}".format(stat_titles[i]),fontsize=10)
  axes[i+4].set_xlabel(stat_titles[i],fontsize=10)
  axes[i+4].set_ylabel("probability",fontsize=10)
  axes[i+4].legend()
  figs[i+4].savefig("../results/inspection_figures/SNR_{:s}.png".format(savefig_labels[i]))
plt.show()

# plot single-figure histograms
matplotlib.rcParams.update({'font.size' : 14})

(fig1,axes1) = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
(fig2,axes2) = plt.subplots(nrows=2,ncols=2,figsize=(10,10))

fig1.suptitle("Integrated Pulse Profile",fontsize=20,y=0.97)
fig2.suptitle("DM-SNR Curve",            fontsize=20,y=0.97)

for i in range(2):
  for j in range(2):
    # IPP histogram
    axes1[i,j].hist(IPP[i+j][idx_nonpulsar],bins=150,label="non-pulsars")
    axes1[i,j].hist(IPP[i+j][idx_pulsar],   bins= 50,label="pulsars")
    axes1[i,j].set_title("distribution of {:s}".format(stat_titles[i+j]),fontsize=16)
    axes1[i,j].set_xlabel(stat_titles[i+j],fontsize=16)
    axes1[i,j].set_ylabel("frequency",   fontsize=16)
    axes1[i,j].legend()
    
    # SNR histogram
    axes2[i,j].hist(SNR[i+j][idx_nonpulsar],bins=150,label="non-pulsars")
    axes2[i,j].hist(SNR[i+j][idx_pulsar],   bins= 50,label="pulsars")
    axes2[i,j].set_title("distribution of {:s}".format(stat_titles[i+j]),fontsize=16)
    axes2[i,j].set_xlabel(stat_titles[i+j],fontsize=16)
    axes2[i,j].set_ylabel("frequency",   fontsize=16)
    axes2[i,j].legend()

fig1.tight_layout()
fig2.tight_layout()

fig1.subplots_adjust(top=0.9)
fig2.subplots_adjust(top=0.9)


fig1.savefig("../results/inspection_figures/IPP.png")
fig2.savefig("../results/inspection_figures/SNR.png")
plt.show()

