import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dbscan_inner import dbscan_inner
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

sys.path.append("../FINDER/Code/")
from finder import Finder
from GridSampler import GridSampler

N_goal = 10000

filename = sys.argv[1]
filename = filename[:-4]
# filename = "../MikeData/EGFR-P1-ATTO655_cell_2_MMImages.ome_locs_render_al_linked1sigma"


XC = np.loadtxt(filename + ".txt")

N = len(XC)

print("Loaded " + str(len(XC)) + " points")

n = np.int(np.sqrt(N / N_goal))
print("Splitting into " + str(n) + " x " + str(n) + " boxes.")

xy_min = np.min(XC, axis=0)
xy_max = np.max(XC, axis=0)
dxy = (xy_max - xy_min) / n

for i0 in np.arange(n):
    for i1 in np.arange(n):

        # Filter
        x_min_i = xy_min[0] + i0 * dxy[0]
        x_max_i = xy_min[0] + (i0 + 1) * dxy[0]

        y_min_i = xy_min[1] + (i1) * dxy[1]
        y_max_i = xy_min[1] + (i1 + 1) * dxy[1]

        print("i0 = " + str(i1) + " , i1 = " + str(i1))
        print(x_min_i)
        print(x_max_i)

        print(y_min_i)
        print(y_max_i)
        mark = (
            (XC[:, 0] >= x_min_i)
            & (XC[:, 0] < x_max_i)
            & (XC[:, 1] >= y_min_i)
            & (XC[:, 1] < y_max_i)
        )

        print(len(XC[mark, :]))

        # Run
        t_start = time.time()
        FD = Finder()
        FD.fit(XC[mark, :])
        delta_t = time.time() - t_start
        print(
            "Computed in "
            + str(np.round(delta_t, 2))
            + " seconds for "
            + str(len(XC))
            + " points"
        )
#
# with open(filename+'_Finder1d.pickle','wb') as handle:
#         pickle.dump({'FD':FD}, handle,protocol=pickle.HIGHEST_PROTOCOL)

# np.savetxt(filename+"_labels.txt",FD.labels,fmt="%d",header="FINDER label, computation time = "+str(np.round(delta_t,2))+" seconds for "+str(len(XC))+" points");
