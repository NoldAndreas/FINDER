#!/usr/bin/python
import sys

import numpy as np

filename = sys.argv[1]
outputfolder = sys.argv[2]


XC = np.loadtxt(filename)

N_x = 1000
XC_min = np.min(XC, axis=0)
XC_max = np.max(XC, axis=0)

# have N points in the x-dimension
N_y = np.int(N_x * (XC_max[1] - XC_min[1]) / (XC_max[0] - XC_min[0]))

print("heat map with dimensions " + str(N_x) + " x " + str(N_y))

xedges = np.linspace(XC_min[0], XC_max[0], N_x + 1)
yedges = np.linspace(XC_min[1], XC_max[1], N_y + 1)

H, xedges, yedges = np.histogram2d(XC[:, 0], XC[:, 1], bins=(xedges, yedges))

X, Y = np.meshgrid(
    (xedges[1:] + xedges[:-1]) / 2, (yedges[1:] + yedges[:-1]) / 2
)

np.savetxt(outputfolder + filename[:-4] + "_heatmap_D.txt", H, fmt="%f")
np.savetxt(outputfolder + filename[:-4] + "_heatmap_X.txt", X, fmt="%f")
np.savetxt(outputfolder + filename[:-4] + "_heatmap_Y.txt", Y, fmt="%f")

# sns.heatmap(H);
# plt.savefig(outputfolder+filename[:-4]+"_heatmap.pdf");
