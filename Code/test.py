#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:11:46 2021

@author: andreas
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import time
import seaborn as sns
import sys
from Finder_1d import Finder_1d
from GridSampler import GridSampler
import scipy.stats


#filename = "../../MikeData/EGFR-P1-ATTO655_cell_2_MMImages.ome_locs_render_al_linked1sigma"
filename = "../../MikeData/XC"
XC = np.loadtxt(filename+".txt");

gs = GridSampler(XC,N_goal=2000);
cls_dist = gs.sampleGrid(algo='FINDER');