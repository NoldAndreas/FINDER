#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:10:20 2020

@author: andreas
"""

import sys
sys.path.append("Modules/")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import seaborn as sns

from Geometry_Grid import Geometry_Grid
from Finder_1d import Finder_1d
from Definitions import get_datafolder
basefolder = get_datafolder()

plt.rcParams['axes.facecolor'] = 'w'

#****************************
# Parameters
threshold            = 10
points_per_dimension = 15; #vary for Fig. S19
#****************************

def PlotScatter(labels,XC,ax=[],showScaleBar=False,showBorder=False):
 
       
        # Get correctly detected:
        correct_detected = np.ones_like(labels);        
        if(ax == []):        
            fig,ax = plt.figure();
        mark = (labels==-1);
        sns.scatterplot(x=XC[mark,0],y=XC[mark,1],color='grey',alpha=0.2,ax=ax);
        mark = (labels>=0);
        sns.scatterplot(x=XC[mark,0],y=XC[mark,1],hue=labels[mark],palette='Set1',
                        size=0.2,style=-1*correct_detected[mark],legend=False,ax=ax);
        ax.set_aspect('equal');

        x_0 = 0;
        y_0 = np.min(XC[:,1]) - 80;        
        if(showScaleBar):            
            ax.plot([x_0,x_0+100],[y_0,y_0],'k')
            ax.annotate('$100nm$',(x_0+50,y_0+10),fontsize='large',ha='center');         
        else:
            ax.plot([x_0,x_0+100],[y_0,y_0],'w')
        
        ax.set_aspect(1);
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.axis('off');

        if(ax==[]):
            plt.show();


for dbscanType in ['dbscan','DbscanLoop']:
    for name_idx in ["FigS3","FigS4"]:
        name = str(name_idx)+'_'+dbscanType
        
            
        if(name_idx == "FigS4"):
            params = {'n_side':5,
                      'seed':1,
                      'Delta_ratio':.8,
                      'noise_ratio':1.,
                      'unit_type':'Clusters_DNA_1mers'};#"Clusters_DNA_1mers";#"Clusters_Neuron";
        elif(name_idx == "FigS3"):
            params = {'n_side':5,
                      'seed':1,
                      'Delta_ratio':0.8,
                      'noise_ratio':1.5,
                      'unit_type':'Clusters_Neuron'};#"Clusters_DNA_1mers";#"Clusters_Neuron";
        
        
      
    
        #****************************
        now             = datetime.now() 
        date_time       = now.strftime("%Y_%m_%d_%H_%M_%S")
        filename_dataframe = "Results_"+date_time+".txt"
        #basefolder = "Results/";
        
                
        G = Geometry_Grid(basefolder,params['unit_type'],
                          n_side=params['n_side'],
                          Delta_ratio=params['Delta_ratio'],
                          noise_ratio=params['noise_ratio'])
        G.GeneratePoints(params['seed'])
        
        #Test: What does testset look like?
        G.PlotScatter(basefolder+name+"_Groundtruth.pdf");        
        XC = G.XC;        
        FD      = Finder_1d(algo=dbscanType,threshold=threshold,points_per_dimension=points_per_dimension);#,points_per_dimension=20);
        labels  = FD.fit(XC);       
        
        
        #*********************************************
        threshold   = FD.threshold; 
        sigmas      = np.asarray(FD.data['sigma'])
        sigma_opt   = FD.selected_parameters['sigma']
        index_opt   = np.where(sigmas==sigma_opt)[0][0]
        
        fig,axs = plt.subplots(1,7,figsize=(14,3)); 
        for i,idx_shift in enumerate([-3,-2,-1,0,1,2,3]):
            if(index_opt+idx_shift < 0):
                axs[i].axis('off')
                continue
            if(index_opt+idx_shift >= len(sigmas)):
                axs[i].axis('off')
                continue;            
            sigma_  = sigmas[index_opt+idx_shift]
            labels_  = FD.ComputeClusters(sigma_,threshold,XC);        
            if(i==0):
                PlotScatter(labels_,XC,ax=axs[i],showScaleBar=False)
            else:
                PlotScatter(labels_,XC,ax=axs[i],showScaleBar=False)
            if(idx_shift == 0):
                axs[i].set_title('eps = '+str(np.round(sigma_,1)), backgroundcolor= 'silver');
            else:
                axs[i].set_title('eps = '+str(np.round(sigma_,1)));
        plt.text(0.1, 0.85, 'd', fontsize=14, transform=plt.gcf().transFigure)            
        plt.savefig(basefolder+name+"_shifts.pdf",bbox_inches="tight");
        
        #*********************************************
        CD_sorted = np.sort(dist.squareform(dist.pdist(XC)),axis=1);      
        sigma_min = np.quantile(CD_sorted[:,threshold],0.1);
        sigma_max = np.quantile(CD_sorted[:,threshold],0.9);    
        #*********************************************
    
    
        fig,axs = plt.subplots(1,1,figsize=(5,5)); 
        ax = axs;
        sns.lineplot(ax=ax,data=FD.data,x='sigma',y='similarityScore');
        sns.scatterplot(ax=ax,data=FD.data,x='sigma',y='similarityScore');
        ax.axvline(sigma_min,c='r');
        ax.axvline(sigma_max,c='r');
        ax.axvline(sigma_opt,c='g');
        #plt.annotate('Selected value', (sigma_opt,FD.phasespace['similarityScore'][index_opt]))
        trans = ax.get_xaxis_transform()
        plt.text(sigma_opt, .5, 'Selected value', transform=trans,rotation=90)
        ax.set_xlabel('Epsilon');
        ax.set_ylabel('Similarity score');
        
        plt.savefig(basefolder+name+"_single_ptsPerDim_"+str(points_per_dimension)+ ".pdf",bbox_inches="tight");
        print("Figure saved in : " + basefolder+name+"_single_ptsPerDim_"+str(points_per_dimension)+ ".pdf");
    
    
        #*********************************************
        fig,axs = plt.subplots(1,3,figsize=(14,5)); 
        fig.tight_layout(pad=3.0)
        ax = axs[0];
        PlotScatter(G.labels_groundtruth,XC,ax=ax,showScaleBar=True);
        
        
        #*********************************************
        ax = axs[1];
        
        nPt_distance = CD_sorted[:,threshold];
        ax.plot(np.arange(len(nPt_distance)),np.sort(nPt_distance));
        #sns.distplot(CD_sorted[:,threshold],ax=ax);
        ax.axhline(sigma_min,c='r');
        ax.axhline(sigma_max,c='r');
        ax.set_ylabel('10-point distance [nm]');
        ax.set_xlabel('Point index (ordered)')
        
        
        #*********************************************
        ax = axs[2];
        sns.lineplot(ax=ax,data=FD.data,x='sigma',y='similarityScore');
        sns.scatterplot(ax=ax,data=FD.data,x='sigma',y='similarityScore');
        ax.axvline(sigma_min,c='r');
        ax.axvline(sigma_max,c='r');
        ax.axvline(sigma_opt,c='g');
        #plt.annotate('Selected value', (sigma_opt,FD.phasespace['similarityScore'][index_opt]))
        trans = ax.get_xaxis_transform()
        plt.text(sigma_opt, .5, 'Selected value', transform=trans,rotation=90)
        ax.set_xlabel('Epsilon');
        ax.set_ylabel('Similarity score');
        
        plt.text(0.06, 0.95, 'a', fontsize=14, transform=plt.gcf().transFigure);
        plt.text(0.35, 0.95, 'b', fontsize=14, transform=plt.gcf().transFigure);
        plt.text(0.65, 0.95, 'c', fontsize=14, transform=plt.gcf().transFigure);      
        plt.savefig(basefolder+name+"_ptsPerDim_"+str(points_per_dimension)+ ".pdf",bbox_inches="tight");
        print("Figure saved in : " + basefolder+name+"_ptsPerDim_"+str(points_per_dimension)+ ".pdf");
    
        
