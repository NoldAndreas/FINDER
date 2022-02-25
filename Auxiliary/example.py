import numpy as np
import time
import sys
import os
from Finder_1d import Finder_1d
from sklearn.cluster import DBSCAN

data_dir = '/home/pietro/Documents/Mainz/Project_1_Andreas/Data_AnalysisOrganized'

# filename = data_dir + '/TTX_control_new/Input/AHA_1_MMImages.ome_locs_render_driftcor_filter_render_pix0.11X6fr20_rois'
filename = data_dir + "/TTX_24hr_2/Input/AHA_1_MMStack_Pos0.ome_locs_render_driftcor_filter_render_pix0.12X6fr20_rois"

XC = np.loadtxt(filename + ".txt")[:, :2]
print("Loaded " + str(len(XC)) + " points")

# # REGULAR DB SCAN
# min_samples = 5
# eps = 0.6
# t1 = time.time()
# DB = DBSCAN(eps=eps,min_samples=min_samples).fit(XC);
# labels_DBSCAN = DB.labels_;
# delta_t = time.time()-t1;
# print("Computed in "+str(np.round(delta_t,2))+" seconds for "+str(len(XC))+" points");
#
# np.savetxt(filename+"_labels.txt",labels_DBSCAN,fmt="%d",header="DBSCAN label, computation time = "+str(np.round(delta_t,2))+" seconds for "+str(len(XC))+" points");


print("----------STARTING FINDER---------")
FD = Finder_1d()
FD_labels = FD.fit(XC[:])


