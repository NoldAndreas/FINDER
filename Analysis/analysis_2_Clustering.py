"""performs clustering in given window"""

import sys
sys.path.append("Modules/")

import os
import numpy as np
import h5py
import json
from datetime import datetime
from ClustersInOutCell import ClustersInOutCell
from Definitions import get_datafolder
#filename      = sys.argv[1];
#OUTPUTFOLDER  = sys.argv[2];

OUTPUTFOLDER = get_datafolder(os.path.join('Data_AnalysisOrganized','Fig3_3mers','Output'))


points_per_dimension = 25
skipSimilarityScore = True
N_SELECT = 4000

assert os.path.exists(OUTPUTFOLDER) , "OUTPUTFOLDER does not exist"

def load_points(filename):
    """Loads x-y-coordinates from .txt or .hdf5-file"""
    #Check if file exists
    assert os.path.isfile(filename), f"File {filename} not found"

    if filename[-3:] == 'txt':
        xc = np.loadtxt(filename)
    elif filename[-4:] == 'hdf5':
        f = h5py.File(filename, 'r')
        dset = f['locs']
        xc = np.stack((dset["x"], dset["y"])).T
    print(str(len(xc)) + ' points loaded from '+filename)
    return xc

def define_window(xc_incell, xc_outcell, n_select, outputfolder):
    """applies select_points() on incell and outcell points and saves congiguration """

    #get timestamp for window
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y_%m_%d__%H_%M_%S")
    outputfolder_window = os.path.join(outputfolder,date_time)+os.path.sep
    if not os.path.exists(outputfolder_window):
        os.makedirs(outputfolder_window)

    #Select windows
    xc_incell_w, xy_center_incell, r_incell = select_points(xc_incell, n_select)
    xc_outcell_w, xy_center_outcell, r_outcell = select_points(xc_outcell, n_select)

    #save points
    np.savetxt(outputfolder_window+"X_incell_window.txt", xc_incell_w, fmt="%f")
    np.savetxt(outputfolder_window+"X_outcell_window.txt", xc_outcell_w, fmt="%f")

    #save PARAMETERFILE
    parameters_window = {'xy_center_incell':(np.float64(xy_center_incell)).tolist(),\
                         'xy_center_outcell':(np.float64(xy_center_outcell)).tolist(),\
                         'r_incell':np.float64(r_incell),\
                         'r_outcell':np.float64(r_outcell),\
                         'N_select':(n_select)}
    with open(outputfolder_window+'parameters_window.json', 'w') as fp:
        json.dump(parameters_window, fp, indent=4)

    return outputfolder_window

def select_points(xc, n_select):
    """selects the closest n_select points of a random point of xc"""
    #xymin,xymax = np.min(XC,axis=0),np.max(XC,axis=0);

    #Choose random point in area:
    xy_center = xc[np.random.randint(len(xc)), :]
    print('Center = '+str(xy_center))
    rs = np.linalg.norm(xc-xy_center, axis=1)

    if n_select < len(xc):
        r = np.sort(rs)[n_select]
    else:
        r = np.sort(rs)[-1]

    return xc[rs < r], xy_center, r




XC_INCELL = load_points(os.path.join(OUTPUTFOLDER,'X_incell.txt'))
XC_OUTCELL = load_points(os.path.join(OUTPUTFOLDER,'X_outcell.txt'))

OUTPUTFOLDER_WINDOW = define_window(XC_INCELL, XC_OUTCELL,\
                        N_SELECT, OUTPUTFOLDER)

CB = ClustersInOutCell(OUTPUTFOLDER_WINDOW, points_per_dimension=points_per_dimension)
CB.GetClusterings_InOutCell(skipSimilarityScore=skipSimilarityScore)
CB.GetReferenceClustering()
