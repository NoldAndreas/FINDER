#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:04:15 2020

@author: andreas
"""

#from Geometry_Path import Geometry_Path
from Geometry_Grid import Geometry_Grid
from Basefolder import basefolder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



for library_type in ['Clusters_Neuron','Clusters_DNA_1mers']:
    
    G = Geometry_Grid(basefolder,library_type,
                                  n_side=0,
                                  Delta_ratio=1,
                                  noise_ratio=0);
    
    template_clusters = G.template_clusters;
    
    sizes = [];
    for cl in G.template_clusters:
        sizes.append(len(cl));
        
    
    
    fig,axs = plt.subplots(1,2,figsize=(10,5));
    
    ax = axs[0];
    sns.histplot(sizes,bins=20,ax=axs[0]);#,norm_hist=True);
    axs[0].axvline(np.mean(sizes), color='r', linestyle='--',linewidth=2)
    
    ax.set_yticks(np.arange(6));
    ax.set_xlim(0,np.max(sizes)*1.05);
    ax.set_xlabel('Points per cluster');
    ax.set_ylabel('Number of clusters');
    
    ax = axs[1];
    L = 1.5*G.GetTypicalDiameter_of_templateClusters();
    alpha = 0.5;
    
    XC = G.template_clusters[0];
    sns.scatterplot(x=XC[:,0],y=XC[:,1],color='grey',alpha=alpha,ax=axs[1]);
    
    XC = G.template_clusters[1] + [L,0];
    sns.scatterplot(x=XC[:,0],y=XC[:,1],color='grey',alpha=alpha,ax=axs[1]);
    
    XC = G.template_clusters[2] + [L,L];
    sns.scatterplot(x=XC[:,0],y=XC[:,1],color='grey',alpha=alpha,ax=axs[1]);
    
    
    XC = G.template_clusters[3] + [0,L];
    sns.scatterplot(x=XC[:,0],y=XC[:,1],color='grey',alpha=alpha,ax=axs[1]);
    
    
    ax.set_aspect('equal');
    
    x_0 = L/2;
    y_0 =  - L/2;
    ax.plot([x_0-50,x_0+50],[y_0,y_0],'k')
    ax.annotate('$100nm$',(x_0,y_0+10),fontsize='large',ha='center');         
    ax.set_aspect(1);
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.axis('off');
    
    print('Larges cluster diameter: '+str(G.GetTypicalDiameter_of_templateClusters()))
    
    plt.savefig(basefolder + 'FigS1_Library_'+library_type+'.pdf',bbox_inches="tight");
    print('Figure saved in: '+basefolder+'FigS1_Library_'+library_type+'.pdf');