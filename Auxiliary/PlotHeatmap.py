#!/usr/bin/python
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

filename = sys.argv[1]

H = np.loadtxt(filename[:-4] + "_heatmap_D.txt")
X = np.loadtxt(filename[:-4] + "_heatmap_X.txt")
Y = np.loadtxt(filename[:-4] + "_heatmap_Y.txt")

plt.contourf(X, Y, H, vmax=100, vmin=0)
plt.savefig(filename[:-4] + "_heatmap.pdf")
