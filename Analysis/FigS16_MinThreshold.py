#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:29:28 2020

@author: andreas
"""
import sys
sys.path.append("Modules/")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from Definitions import get_datafolder
basefolder = get_datafolder()

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


dict_algo_names_ = {"OPTICS":"OPTICS",
                    "dbscan":"DBSCAN",
                    "CAML_07VEJJ":"CAML (07VEJJ)",
                    "CAML_87B144":"CAML (87B144)",
                    "FINDER_1D_loop":"FINDER",
                    "FINDER_1D":"FINDER  with DBSCAN"
                    };

basefolder = basefolder+'/Results_Fig4/';

filenamesList = ['Results_5','Results_4','Results_0','Results_3','Results_1','Results_2'];
x_names       = ['Delta_ratio','noise_ratio','N_clusters','Delta_ratio','noise_ratio','N_clusters'];
xs_to_display = [1.0,1.0,25,1.0,1.0,25];

def clusterEvaluation(labels,labels_groundTruth,min_overlap_per_ref):
                         
    no_clusters         = np.max(labels) +1;
    no_clusters_gT      = np.max(labels_groundTruth) +1;    
    clusters_correct    = -np.ones((no_clusters,),dtype=np.int);
    
    clusters     = [np.where(labels == i)[0] for i in np.arange(no_clusters)];
    clusters_ref = [np.where(labels_groundTruth == i)[0] for i in np.arange(no_clusters_gT)];        
            
    for i,cl in enumerate(clusters):            
        #center = np.mean(XC[cl,:],axis=0);
        coverage = 0;
        idx_chosen = -1;
        
        for ii,cl_ref in enumerate(clusters_ref):
            
            cl_and_clRef = np.intersect1d(cl,cl_ref);
            
            #center_ref = np.mean(XC[cl_ref,:],axis=0);
            clusters_correct[i] = -1;
            if(len(cl_and_clRef) > min_overlap_per_ref*len(cl_ref)):              
            #if(np.linalg.norm(center_ref-center)<0.05):

                #remove cl_ref from clusters_ref
                #
#                    if(ii in list(clusters_correct)):                                                                    
#                        print("DOUBLE ASSIGNMENT");
#                        raise;
#                        continue;
                
                if(len(cl_and_clRef)/len(cl) > coverage):
                    idx_chosen = ii;
                    coverage   = len(cl_and_clRef)/len(cl);
        
        clusters_ref =  [clusters_ref[i] for i in np.arange(len(clusters_ref)) if (i!=idx_chosen)]
        clusters_correct[i] = idx_chosen;                    
                
    return clusters_correct        
        
def analyseRatios(cl_,ratios):
    
    labels_groundtruth = cl_.Geometry.labels_groundtruth;
    labels             = cl_.labels;
    
    

    true_pos = np.zeros_like(ratios);
    false_pos = np.zeros_like(ratios);
    
    for i,ratio_ in enumerate(ratios):
        clusters_correct = clusterEvaluation(labels,labels_groundtruth,ratio_);
        
        #false_negative_clusters = np.asarray([a for a in all_groundtruth if (not (a in list(clusters_correct)))]);
        false_pos[i] = np.sum(clusters_correct == -1)/cl_.Geometry.N_clusters;
        true_pos[i]  = np.sum(clusters_correct != -1)/cl_.Geometry.N_clusters;

    return true_pos,false_pos;


#************************
# Set up figure
#************************
gs_kw = dict(width_ratios=[1,1,1,1,1,1], height_ratios=[4,4,1]);
fig,axs = plt.subplots(nrows=3,ncols=6,gridspec_kw=gs_kw,figsize=(14,8));#gridspec_kw=gs_kw

gs = axs[-1, -1].get_gridspec()
# remove the underlying axes
for ax in axs[-1,:]:
    ax.remove()
ax_legend = fig.add_subplot(gs[-1,:])


#***********************************
# Load and plot results
#***********************************
for i in np.arange(6):
    
    filename = filenamesList[i];
    x_name = x_names[i];
    x_to_display = xs_to_display[i];
 
    #***************************
    # Load data
    #***************************
    filename_pickle = basefolder+filename+'.pickle'
    all_data = []
    infile = open(filename_pickle, "rb")
    while 1:
        try:            
            pp = pickle.load(infile);
            if(x_name == 'Delta_ratio'):
                if(pp.Geometry.parameters['Delta_ratio'] != x_to_display):
                    continue;                
            elif(x_name == 'noise_ratio'):            
                if(pp.Geometry.noise_ratio != x_to_display):
                    continue;
            elif(x_name == 'N_clusters'):  
                if(pp.Geometry.N_clusters != x_to_display):
                    continue;
            
            all_data.append(pp)
        except (EOFError, pickle.UnpicklingError):
            break
    infile.close();
    

    
    #***************************
    # Load data
    #***************************
    data_eval = {'ratio':[], 
                  'algo':[],
                  'true_pos':[],
                  'false_pos':[],
                  x_name:[]}; 
    ratios = np.linspace(0.0,1.0,20);
    
    
    for cl_ in all_data:
        true_pos_,false_pos_ = analyseRatios(cl_,ratios);   
        for ii,tp in enumerate(true_pos_):
            data_eval['algo'].append(cl_.algo);
            data_eval['true_pos'].append(tp);
            data_eval['false_pos'].append(false_pos_[ii]);            
            data_eval['ratio'].append(ratios[ii]);
            
            if(x_name == 'Delta_ratio'):            
                data_eval['Delta_ratio'].append(cl_.Geometry.parameters['Delta_ratio']);
            elif(x_name == 'noise_ratio'):            
                data_eval['noise_ratio'].append(cl_.Geometry.noise_ratio);
            elif(x_name == 'N_clusters'):            
                data_eval['N_clusters'].append(cl_.Geometry.N_clusters);   


    df = pd.DataFrame(data=data_eval);
    mark = (np.asarray(data_eval[x_name]) == x_to_display);

    
    sns.lineplot(data=df[mark],x='ratio',y='true_pos',hue='algo',ax=axs[0,i]);
    sns.lineplot(data=df[mark],x='ratio',y='false_pos',hue='algo',ax=axs[1,i]);    
    
    for ir in [0,1]:
        axs[ir,i].legend([],[], frameon=False);        
        if(i>0):
            axs[ir,i].set_ylabel("");
            axs[ir,i].set_yticks([]);
        else:
            if(ir==0):
                axs[ir,i].set_ylabel("True positives (ratio)");
            else:
                axs[ir,i].set_ylabel("False positives (ratio)");
        axs[ir,i].set_xlim(0,1);
    axs[0,i].set_ylim(0,1);        
    axs[1,i].set_ylim(0,1.5);        
    axs[0,i].set_xlabel("");
    axs[0,i].set_xticks([]);    
    axs[1,i].set_xlabel("Min overlap (ratio)");

#************************
# Add labels
#************************
pos_ = np.linspace(0.11,0.79,6)
plt.text(pos_[0], 0.9, 'a', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[1], 0.9, 'b', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[2], 0.9, 'c', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[3], 0.9, 'd', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[4], 0.9, 'e', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[5], 0.9, 'f', fontsize=14, transform=plt.gcf().transFigure)

d__ = +0.025;
plt.text(pos_[0]+d__, 0.9, 'High overlap', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[1]+d__, 0.9, 'High noise', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[2]+d__, 0.9, 'Unstructured', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[3]+d__, 0.9, 'High overlap', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[4]+d__, 0.9, 'High noise', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[5]+d__, 0.9, 'Unstructured', fontsize=14, transform=plt.gcf().transFigure)


plt.text(0.25, 0.96, 'Low density clusters', fontsize=14, transform=plt.gcf().transFigure)
plt.text(0.65, 0.96, 'High density clusters', fontsize=14, transform=plt.gcf().transFigure)

#************************
# Add legend
#************************
lines, labels = axs[0,0].get_legend_handles_labels();
ax_legend.set_xticks([]);
ax_legend.set_yticks([]);
ax_legend.axis('off');

labels_properNames = [dict_algo_names_[l] for l in labels];
ax_legend.legend(lines, labels_properNames, loc = 'center',ncol=6,facecolor='white',frameon=False);#framealpha=1

#plt.subplots_adjust(wspace=0.35);    
plt.savefig(basefolder + "FigS16_ThresholdVariation.pdf",bbox_inches="tight");
print("File saved in : "+basefolder + "FigS16_ThresholdVariation.pdf");