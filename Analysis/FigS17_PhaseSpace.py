#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 22:26:43 2021

@author: andreas
"""
import os.path
import sys


import numpy as np
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
import seaborn as sns
from DbscanLoop import DbscanLoop
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import h5py
from Finder import Finder

from Definitions import get_datafolder

base_folder = os.path.dirname(os.getcwd())
data_folder = os.path.join(os.path.dirname(os.getcwd()), 'data_sources')

sys.path.append("Modules/")
#style.use('ClusterStyle')

#****************************
# Parameters
#****************************
k          = 10
minPts_min = 3
minPts_max = 17
n          = 20

colors_list_dark  = ['#194b8f','#da6500','#1c641c']
colors_list_light = ['#4aafe0','#ffa75b','#4bce4b'] 

hue_order = ['FINDER_1D_loop','FINDER_1D','dbscan','CAML_87B144','CAML_07VEJJ','OPTICS']

#e8a358
my_pal = {'CAML_07VEJJ':'#eabe8e',\
          'CAML_87B144':'#d67d1d',\
          'FINDER_1D_loop':'#701ac0',\
          'FINDER_1D':'#af6eeb',\
          'dbscan':'dimgrey',\
          'OPTICS':'lightgrey',\
        };  
    
algo = "DbscanLoop";


do_computation = False

#****************************
# Load points
#****************************
for case in ["neuron","ttx"]:
    if do_computation == True:
        if(case == "neuron"):
            XC      = np.loadtxt(data_folder + '/TemplateClusters/NeuronData/dendrite_example_Cell1_GluA2_40I_ROI1_1_MMStack_Pos0.ome_locs_render_driftCorr_filter_render_pix.6fr20_picked2_picked3.txt');
        elif(case == "ribosome"):
            XS      = np.loadtxt(data_folder + '/TemplateClusters/RibosomeData/40s.txt');
            mark    = (XS[:,0]>360)*(XS[:,0]<380)*(XS[:,1]>260)*(XS[:,1]<280);
            XC      = XS[mark,:];
        elif(case=="ttx"):
            datascale   = 158;
            filename = 'TemplateClusters/ProteinData_ttx_1hr_2/AHA_2_MMStack_Pos0.ome_locs_render_driftcor_filter_render_pix0.02X6f20.hdf5';
            f        = h5py.File(data_folder + filename, 'r')
            dset     = f['locs'];
            XC       = np.stack((dset["x"],dset["y"])).T
            XC        = datascale*XC;
            XC        = np.unique(XC,axis=0);

            lims_x = [37000,43000];
            lims_y = [4000,10000];

            markPlot = (XC[:,0]>lims_x[0])*(XC[:,0]<lims_x[1])*(XC[:,1]>lims_y[0])*(XC[:,1]<lims_y[1]);
            XC       = XC[markPlot,:];


        print("Loaded "+str(len(XC))+' localizations');
            #****************************
        # Get Sigma boundaries
        #****************************


        # initialize model
        neigh = NearestNeighbors(n_neighbors=k+1, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(XC);
        dist_, ind = neigh.kneighbors(XC);

        # We have no use of indices here
        # dist is a 2 dimensional array of shape (10000, 9) in which each row is list of length 9. This row contain distances to all 9 nearest points. But we need distance to only 9th nearest point. So
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]

        #CD_sorted = np.sort(dist.squareform(dist.pdist(XC)),axis=1);
        sigma_min = np.quantile(nPt_distance,0.2);
        sigma_max = np.quantile(nPt_distance,0.99);

        #****************************
        #Get Vector of thresholds an sigmas:
        #****************************

        #thresholds = np.unique((np.exp(np.linspace(np.log(minPts_min),np.log(minPts_max),n))).astype(int));
        sigmas     = np.unique(np.exp(np.linspace(np.log(sigma_min),np.log(sigma_max),n)));
        thresholds = np.arange(minPts_min,minPts_max+1);

        # Compute phasespace
        noClustersMatrix = np.zeros((len(sigmas),len(thresholds)),dtype=int);
        data = {'sigma':[],
                'minPts':[],
                'noClusters':[]};

        print("Computing..")
        for i1,sigma in enumerate(sigmas):
            ps_t = []
            for i2,threshold in enumerate(thresholds):
                if(algo=="dbscan"):
                    DB          = DBSCAN(eps=sigma,min_samples=threshold).fit(XC);
                elif(algo=="DbscanLoop"):
                    DB          = DbscanLoop(eps=sigma,min_samples=threshold).fit(XC);
                labels      = DB.labels_;
                noClustersMatrix[i1,i2] = np.max(labels)+1;

        FD      = Finder(algo='DbscanLoop');
        labels  = FD.fit(XC);

    else:
        if case == "neuron":
            pkl_path = os.path.join(data_folder,"phase_spaces","neuron_phasespace.pkl")
            PS = pd.read_pickle(pkl_path)
            neuron_name = "dendrite_example_Cell1_GluA2_40I_ROI1_1_MMStack_Pos0.ome_locs_render_driftCorr_filter_render_pix.6fr20_picked2_picked3.txt"
            neuron_path = os.path.join(data_folder,"TemplateClusters","NeuronData", neuron_name)
            XC = np.loadtxt(neuron_path)

            no_clusters = []

            for i, row in PS.iterrows():
                no_clusters.append(row["no_clusters"])

            noClustersMatrix = (np.array(no_clusters).reshape(15,-1)) + 1
            sigmas = np.unique(PS["sigma"])
            thresholds = np.unique(PS["threshold"])

        if case == "ttx":
            pkl_path = os.path.join(data_folder,"phase_spaces","protein_ttx_1hr_2_phasespace.pkl")
            if not os.path.isfile(pkl_path):
                print("The file'protein_ttx_1hr_2_phasespace.pkl' was not uploaded in this directory because it is too large.")
                print("Please contact us if you need to use it")

            PS = pd.read_pickle(pkl_path)
            ttx_name = "AHA_2_MMStack_Pos0.ome_locs_render_driftcor_filter_render_pix0.02X6f20.hdf5"
            filename = os.path.join(data_folder,"TemplateClusters","ProteinData_ttx_1hr_2", ttx_name)
            if not os.path.isfile(filename):
                print("The file for 'ProteinData_ttx_1hr_2' was not uploaded in this directory because it is too large.")
                print("Please contact us if you need to use it")
            f = h5py.File(filename, 'r')
            dset = f['locs'];
            XC = np.stack((dset["x"], dset["y"])).T
            no_clusters = []

            for i, row in PS.iterrows():
                no_clusters.append(row["no_clusters"])

            noClustersMatrix = (np.array(no_clusters).reshape(15,-1)) + 1
            sigmas = np.unique(PS["sigma"])
            thresholds = np.unique(PS["threshold"])

    fig,axs = plt.subplots(figsize=(8,8));
    sns.heatmap(np.flipud(noClustersMatrix.T),xticklabels=np.round(sigmas,2),yticklabels=np.flipud(np.round(thresholds)),annot=True,fmt="d",ax=axs,cbar=False,cmap='Reds');
    sns.color_palette("light:b", as_cmap=True)
    axs.set_aspect('equal');
    axs.set_xlabel('Distance r (nm)');
    axs.set_ylabel('minPts');
    
    plt.savefig(data_folder + "/Results/FigS17_18_a_" + case + "_Phasediagram.pdf", bbox_inches="tight");
    print("File saved in : " + data_folder + "/Results/FigS17_18_a_" + case + "_Phasediagram.pdf");
    
    #*********************
    # Plot examples
    #*********************
    sigmas_idx = [6,9,12,14]
    minPts_idx = [7,0];
    
    fig,axs = plt.subplots(2,4,figsize=(12,8));
    
    #for i1,sigma_idx in enumerate([]):
    for ii,si in enumerate(sigmas_idx):
        for jj,mi in enumerate(minPts_idx):
        
            sigma = sigmas[si];
            minPts = thresholds[mi];
            ax = axs[jj,ii];
            labels   = DbscanLoop(eps=sigma,min_samples=minPts).fit(XC).labels_; 
            
            mark = (labels==-1);
            sns.scatterplot(x=XC[mark,0],y=XC[mark,1],color='grey',alpha=0.2,ax=ax);
            mark = (labels>=0);
            sns.scatterplot(x=XC[mark,0],y=XC[mark,1],hue=labels[mark],palette='deep',
                        size=0.2,legend=False,ax=ax);
            
            ax.set_title('minPts = '+str(minPts)+' \n r = '+str(np.round(sigma,2))+' nm');
            ax.set_aspect(1);
            ax.set_xticks([]);
            ax.set_yticks([]);
            ax.axis('off');
    
    plt.savefig(data_folder + "/FigS17_18_b_" + case + "_ClusteringResults.pdf", bbox_inches="tight");
    print("File saved in : " + data_folder + "/FigS17_18_a_" + case + "_Phasediagram.pdf");