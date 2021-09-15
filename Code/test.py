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


filename = "../../ProteinData_ttx_1hr_2/Analysis_dataWindow_3/dataWindow_3_filtered_signal.txt"

XC = np.loadtxt(filename);

#gs = GridSampler(XC,N_goal=2000);
#cls_dist = gs.sampleGrid(algo='FINDER');

FD = Finder_1d();
FD.fit(XC[:2000]);
