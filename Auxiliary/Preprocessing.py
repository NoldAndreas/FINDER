

import numpy as np
import sys
import os
import h5py
from datetime import datetime
import json

#from PIL import Image
#import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from skimage import measure
from skimage import filters
from skimage.filters import threshold_otsu

from ClusterBasing import ClusterBasing
#filename      = sys.argv[1];
#outputfolder  = sys.argv[2];

#basefolder    = "../../AnalysisDataOrganized/AHA_2_MMStack_Pos0.ome_locs_render_driftcor_filter_render_pix0.02X6f20/";
basefolder    = "../../AnalysisDataOrganized/Mike_190516_EGFR-P1-ATTO655_cell_2_MMImages.ome_locs_render_al_linked1sigma/";
outputfolder  = basefolder+"Output/"

if(not (os.path.exists(outputfolder))):
  os.makedirs(outputfolder);

#*********************************************
# Parameters
#*********************************************

intensity_quantile_cutoff = 0.95;
sigma_gaussian_filter     = 5;
N_x                       = 1000;
N_select                  = 10000;

outputfolder_window       = "";#outputfolder+'2021_09_29__11_32_55/';
#datascale     :158,
#*********************************************

def LoadPoints(filename):
    if(filename[-3:]=='txt'):
        XC = np.loadtxt(filename);
    elif(filename[-4:]=='hdf5'):
        f             = h5py.File(filename, 'r')
        dset          = f['locs'];
        XC            = np.stack((dset["x"],dset["y"])).T;
    print(str(len(XC)) + ' points loaded from '+filename);
    return XC;

def GetImageFromLocalizations(XC):

    XC_min = np.min(XC,axis=0);
    XC_max = np.max(XC,axis=0);

    #have N points in the x-dimension
    N_y    = np.int(N_x*(XC_max[1]-XC_min[1])/(XC_max[0]-XC_min[0]));

    print("heat map with dimensions "+str(N_x)+" x "+str(N_y));

    xedges = np.linspace(XC_min[0],XC_max[0],N_x+1);
    yedges = np.linspace(XC_min[1],XC_max[1],N_y+1);

    H, xedges, yedges = np.histogram2d(XC[:,0],XC[:,1],bins=(xedges, yedges));

    X, Y = np.meshgrid((xedges[1:] + xedges[:-1]) / 2,
                       (yedges[1:] + yedges[:-1]) / 2);

    #np.savetxt(outputfolder+"heatmap_D.txt",H,fmt="%f");
    #np.savetxt(outputfolder+"heatmap_X.txt",X,fmt="%f");
    #np.savetxt(outputfolder+"heatmap_Y.txt",Y,fmt="%f");

    return H,X,Y

def GetIndex(X,Y,XC):

    n_x,n_y     = X.shape[1],X.shape[0];
    dx,dy       = X[1,1]-X[0,0],Y[1,1]-Y[0,0];
    x_min,y_min = X[0,0],Y[0,0];

    x_index = (np.round((XC[:,0]-x_min)/dx)).astype(int)
    y_index = (np.round((XC[:,1]-y_min)/dy)).astype(int)

    x_index = np.minimum(n_x-1,np.maximum(0,x_index));
    y_index = np.minimum(n_y-1,np.maximum(0,y_index));

    return y_index*n_x+x_index

def SelectPoints(XC,N_select):

    xymin,xymax = np.min(XC,axis=0),np.max(XC,axis=0);

    #Choose random point in area:
    xy_center = XC[np.random.randint(len(XC)),:]
    #xy_center = xymin + np.random.rand(2)*(xymax-xymin);
    print('Center = '+str(xy_center));
    rs = np.linalg.norm(XC-xy_center,axis=1)
    r  = np.sort(rs)[N_select];

    return XC[rs<r],xy_center,r;

def DefineWindow(XC_incell,XC_outcell,N_select,outputfolder):

    #get timestamp for window
    now       = datetime.now() # current date and time
    date_time = now.strftime("%Y_%m_%d__%H_%M_%S");
    outputfolder_window = outputfolder + date_time+"/";
    if(not (os.path.exists(outputfolder_window))):
        os.makedirs(outputfolder_window);

    #Select windows
    XC_incell_w ,xy_center_incell,r_incell   = SelectPoints(XC_incell,N_select);
    XC_outcell_w,xy_center_outcell,r_outcell = SelectPoints(XC_outcell,N_select);

    #save points
    np.savetxt(outputfolder_window+"X_incell_window.txt" ,XC_incell_w,fmt="%f");
    np.savetxt(outputfolder_window+"X_outcell_window.txt",XC_outcell_w,fmt="%f");

    #save parameterfile
    parameters_window = {'xy_center_incell' :list(xy_center_incell),\
                         'xy_center_outcell':list(xy_center_outcell),\
                         'r_incell':r_incell,\
                         'r_outcell':r_outcell,\
                         'N_select':N_select};
    with open(outputfolder_window+'parameters_window.json', 'w') as fp:
        json.dump(parameters_window, fp,indent=4);

    return outputfolder_window;

#*********************************************
# Code
#*********************************************
print('*********************************************');
print('Preprocessing..');
print('*********************************************');

if(os.path.isfile(outputfolder+'X_incell.txt') and \
      os.path.isfile(outputfolder+'X_outcell.txt')):
    XC_incell  = LoadPoints(outputfolder+'X_incell.txt');
    XC_outcell = LoadPoints(outputfolder+'X_outcell.txt');
else:
    XC = LoadPoints(basefolder+'Input/XC.hdf5');

    # Step 1: Produce image
    H,X,Y = GetImageFromLocalizations(XC);

    # Step 2 a: Thresholding
    H[H > np.quantile(H,intensity_quantile_cutoff)] = np.quantile(H,intensity_quantile_cutoff);

    # Step 2 b: Gaussian Filtering
    g_img = filters.gaussian(H,sigma=sigma_gaussian_filter);

    # Step 2 c: Otsu's Thresholding
    binary = (g_img > threshold_otsu(g_img));

    # Step 2 d: Segmentation
    labels = measure.label(binary);

    #Step 2 e:
    df_lab = pd.DataFrame();
    df_lab['labels'] = (labels[labels>0]).flatten()
    df_lab['index1'] = df_lab.index
    labG = df_lab.groupby(by='labels');

    label_incell  = (labG.count().idxmax())['index1'];
    label_outcell = 0;

    im_incell = np.zeros_like(H,dtype=np.bool_);
    im_incell[labels==label_incell] = True;

    im_outcell = np.zeros_like(H,dtype=np.bool_);
    im_outcell[labels==label_outcell] = True;

    # Step 3: Save GetImageFromLocalizations
    H_incell_flat  = (im_incell.T).flatten()
    H_outcell_flat = (im_outcell.T).flatten()

    mask_incell  = H_incell_flat[GetIndex(X,Y,XC)];
    mask_outcell = H_outcell_flat[GetIndex(X,Y,XC)];

    XC_incell   = XC[mask_incell];  #==True
    XC_outcell  = XC[mask_outcell]; #==True

    np.savetxt(outputfolder+"X_incell.txt",XC_incell,fmt="%f");
    np.savetxt(outputfolder+"X_outcell.txt",XC_outcell,fmt="%f");

#***************************************************************
# Clustering (one run)
#***************************************************************

if(outputfolder_window==""):
    outputfolder_window = DefineWindow(XC_incell,XC_outcell,N_select,outputfolder);

CB = ClusterBasing(outputfolder_window);
CB.GetClusterings_InOutCell();
CB.GetReferenceClustering();
