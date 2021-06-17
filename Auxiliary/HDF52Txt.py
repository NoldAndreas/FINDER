#!/usr/bin/python
import h5py
import numpy as np
import sys


filename      = sys.argv[1];
f             = h5py.File(filename, 'r')
dset          = f['locs'];
XC            = np.stack((dset["x"],dset["y"])).T;
print(str(len(XC))+" points");
np.savetxt(filename[:-5]+".txt",XC,fmt="%f %f",header="x y");
