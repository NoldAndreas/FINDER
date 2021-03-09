#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:52:06 2020

@author: andreas
"""
from Geometry_Base import Geometry_Base;
import numpy as np

class Geometry_Path(Geometry_Base):

    def __init__(self,basefolder,unitCluster_Library):
        
        super().__init__(basefolder,unitCluster_Library);
        
        self.geometry_name = 'Geometry_Path';
        self.parameters = {"y_scale_noise":1,
                           "y_scale":2,
                           "x_scale":20};
        #'N_SamplePoints":2000}
        
        
    def GeneratePoints(self,N_clusters,N_Noise,seed):
        
        
        np.random.seed(seed);
                
        XC_cluster_centers               = self.__generateSamplePoints(N_clusters);                        
        XC_all,labels,selected_clusters  = self.__positionClusters(XC_cluster_centers);
        
        XC_noise = self.__generateSamplePoints(N_Noise);
                
        XC_all   = np.concatenate((XC_all,XC_noise));
        labels   = np.concatenate((labels,-np.ones((len(XC_noise),),dtype=np.int)));
      

        self.labels_groundtruth = labels;
        self.XC                 = XC_all;
        self.N_clusters         = N_clusters;
        self.N_Noise            = N_Noise;
        self.seed               = seed;
        
       
    def __generateSamplePoints(self,N):
        
        params = self.parameters;
        
        D    = self.GetTypicalDiameter_of_templateClusters();  
                
        y_scale         = params["y_scale"];
        x_scale         = params["x_scale"];
        y_scale_noise   = params["y_scale_noise"];
                
        x_ = np.random.uniform(0,x_scale,N);     
        y_ = y_scale*np.sin(2*3.14*x_/x_scale) +\
                np.random.normal(loc=0.0,scale=y_scale_noise,size=N);
        XC_Sample = D*(np.array([x_,y_])).T;
        
        return XC_Sample;
        
   
    def __positionClusters(self,XC_cluster_centers):
        
        template_clusters = self.template_clusters;
        
        XC_all = np.empty([0,2]);
        labels_ = np.array((),dtype=np.int);
        
        selected_clusters = [];
        
        for idx_cl,cl_c in enumerate(XC_cluster_centers):
            #Select on cluster randomly
            idx = np.random.randint(len(template_clusters));
            #print(idx);
            selected_clusters.append(idx);
            
            XC_cluster_to_include = template_clusters[idx];
            XC_cluster_to_include = XC_cluster_to_include - np.mean(XC_cluster_to_include,axis=0) + cl_c;
            
            XC_all = np.concatenate((XC_all,XC_cluster_to_include));
            labels_ = np.concatenate((labels_,idx_cl*np.ones((len(XC_cluster_to_include),),dtype=np.int)));
            
        return XC_all,labels_,selected_clusters;