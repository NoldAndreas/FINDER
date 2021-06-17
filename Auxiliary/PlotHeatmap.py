#!/usr/bin/python
import numpy as np
import sys
import seaborn as sns;
import matplotlib.pyplot as plt

filename      = sys.argv[1];

H = np.loadtxt(filename[:-4]+"_heatmap_D.txt");
X = np.loadtxt(filename[:-4]+"_heatmap_X.txt");
Y = np.loadtxt(filename[:-4]+"_heatmap_Y.txt");

plt.contourf(X,Y,H,vmax=100,vmin=0);
plt.savefig(filename[:-4]+"_heatmap.pdf");
