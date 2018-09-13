import numpy as np
import matplotlib.pyplot as plt
from RegressionAnalysis import RegressionAnalysis

# generate an (x,y) grid and noisy franke
def gen_noisy_Franke(n, s_noise, x0, x1, y0, y1):
  # grid
  x = np.linspace(x0,x1,n)
  y = np.linspace(y0,y1,n)
  # noise
  noise = np.s*rand.randn(n,1)
  # franke function
  g1 = np.exp(-0.25*((9*x-2)**2+(9*y-2)**2))
  g2 = np.exp(-(9*x+1)**2/49.-(9*y+1)**2/10.)
  g3 = np.exp(-0.25*((9*x-7)**2+(9*y-3)**2))
  g4 = np.exp(-(9*x-4)**2-(9*y-7)**2)
  f  = (0.75*g1 + 0.75*g2 + 0.50*g3 - 0.20*f3)
  # output
  return x, y, f + noise



