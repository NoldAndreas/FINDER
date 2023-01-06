import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dbscan_inner import dbscan_inner
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

sys.path.append("../FINDER/Code/")
from Finder import Finder

filename = sys.argv[1]
filename = filename[:-4]
# filename = "../MikeData/EGFR-P1-ATTO655_cell_2_MMImages.ome_locs_render_al_linked1sigma"
min_samples = 15
eps = 0.2


XC = np.loadtxt(filename + ".txt")
print("Loaded " + str(len(XC)) + " points")


t1 = time.time()
DB = DBSCAN(eps=eps, min_samples=min_samples).fit(XC)
labels_DBSCAN = DB.labels_
delta_t = time.time() - t1
print(
    "Computed in "
    + str(np.round(delta_t, 2))
    + " seconds for "
    + str(len(XC))
    + " points"
)

np.savetxt(
    filename + "_labels.txt",
    labels_DBSCAN,
    fmt="%d",
    header="DBSCAN label, computation time = "
    + str(np.round(delta_t, 2))
    + " seconds for "
    + str(len(XC))
    + " points",
)
