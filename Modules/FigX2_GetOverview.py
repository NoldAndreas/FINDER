#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:46:41 2021

@author: andreas
"""

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

# print (sys.argv[1]);
# filename = sys.argv[1];
foldername = (
    "/Users/andreas/Documents/NoiseRecognizer_WorkingVersion/Polysome_plate/"
)
filename = "40s.txt"


if filename[-3:] == "txt":
    XC = np.loadtxt(foldername + filename)
elif filename[-3:] == "hdf5":
    # filename = '/Users/andreas/Documents/NoiseRecognizer_WorkingVersion/ProteinData_ttx_1hr_2/AHA_2_MMStack_Pos0.ome_locs_render_driftcor_filter_render_pix0.02X6f20.hdf5';
    f = h5py.File(foldername + filename, "r")
    dset = f["locs"]
    XC = np.stack((dset["x"], dset["y"])).T

print(len(XC))
XC = np.unique(XC, axis=0)
# np.savetxt('temp/XC.txt',XC,fmt="%f %f");

print(len(XC))


fig, axs = plt.subplots(1, 1, figsize=(14, 14))
plt.scatter(XC[:, 0], XC[:, 1], s=0.5, alpha=0.5)
axs.set_aspect(aspect=1)
plt.savefig(foldername + "XC.pdf", bbox_inches="tight")
