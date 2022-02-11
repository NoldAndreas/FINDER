#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:51:07 2020

@author: andreas
"""

import numpy as np
from abc import abstractmethod
#import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as dist
import glob
import os

#Geometry base class

class Geometry_Base:
    
    @abstractmethod
    def __init__(self,basefolder,unitCluster_Library):
        
        #Values that have to be set in child classes:
        self.basefolder = basefolder;
        self.geometry_name = [];
        self.XC = [];
        self.N_clusters = [];
        self.N_Noise = [];
        self.labels_groundtruth = [];
        self.parameters = [];
        self.unitCluster_Library = unitCluster_Library;
        self.seed = [];

        self.__loadUnitClusters();
        
    #Methods that need to be set in child classes:
    @abstractmethod        
    def GeneratePoints(self):
        yield None
        
    def __loadUnitClusters(self):       
        data_template_clusters = [];
        folder_ = self.unitCluster_Library;

        template_clusters_paths = os.path.join(self.basefolder, 'TemplateClusters', folder_, 'cluster_*.txt')

        filenamesList = glob.glob(template_clusters_paths);
        for fn in filenamesList:
            XS_i = np.loadtxt(fn, comments="#", delimiter=" ", unpack=False);
            data_template_clusters.append(XS_i); 
        print("Loaded "+str(len(data_template_clusters))+" template clusters..");

        #In the input data, 
        #1 unit corresponds to 158nm. We normalize such that 1 unit = 1 nanometer
        datascale = 158;
        for i,X_cl in enumerate(data_template_clusters):
            data_template_clusters[i] = datascale*(X_cl - np.mean(X_cl,axis=0));
            
        self.template_clusters = data_template_clusters;
        
    def PlotScatter(self,filename):
 
        labels = self.labels_groundtruth;
        XC     = self.XC;
        
        plt.figure();
        mark = (labels==-1);
        sns.scatterplot(x=XC[mark,0],y=XC[mark,1],color='grey',alpha=0.2);
        mark = (labels>=0);
        sns.scatterplot(x=XC[mark,0],y=XC[mark,1],hue=labels[mark],palette='bright',legend=False);
        plt.axis('equal')
        plt.savefig(filename);
        
    def GetTypicalDiameter_of_templateClusters(self):
        D_ = 0;        
        for cl in self.template_clusters:
            d = np.max(dist.pdist(cl));
            if(d>D_):
                D_ = d;
        return D_
    
    def GetTypical_Number_of_points_templateClusters(self):
        Ns = [len(cl) for cl in self.template_clusters];        
        
        return np.mean(np.asarray(Ns));