#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Geometry_Base import Geometry_Base;
import numpy as np
import scipy.spatial.distance as dist


class Geometry_Free(Geometry_Base):

    def __init__(self,basefolder,unitCluster_Library,n_side=2,Delta_ratio=1,noise_ratio=0.):
        
        super().__init__(basefolder,unitCluster_Library);
        
        self.geometry_name = 'Geometry_Free';
        self.parameters = {"Delta_ratio":Delta_ratio}
        self.noise_ratio = noise_ratio;
        
        
    def GeneratePoints(self,seed):
                
        Delta_ratio   = 3;#self.parameters["Delta_ratio"];        

        np.random.seed(seed);

        D    = self.GetTypicalDiameter_of_templateClusters();  
        N_cl = len(self.template_clusters);
        
        XC_all  = np.empty([0,2]);
        labels_ = np.array((),dtype=np.int);
        
        #***********************************
        # Define limits of box
        L_box = np.sqrt(N_cl)*D*Delta_ratio;
        #***********************************
        # Consecutively assign positions 
        for idx,cl in enumerate(self.template_clusters):
            #Get point:
                    
            while(True):
                XC_new = cl+ L_box*np.random.uniform(low=0, high=1, size=(1,2)); 
                
                if(len(XC_all)==0):
                    break;
                if( np.min(dist.cdist(XC_new,XC_all)) > D):
                    break;
           
                
            XC_all = np.concatenate((XC_all,XC_new));
            labels_ = np.concatenate((labels_,idx*np.ones((len(XC_new),),dtype=np.int)));
            
            
        #***********************************
        # Step 2: Add noise to grid
        N_Noise = np.int(self.noise_ratio*len(XC_all));
        x_n = np.random.uniform(low=np.min(XC_all[:,0]), high=np.max(XC_all[:,0]), size=(N_Noise,));
        y_n = np.random.uniform(low=np.min(XC_all[:,1]), high=np.max(XC_all[:,1]), size=(N_Noise,))        
        
        X_noise = np.asarray([x_n,y_n]).T;
        
        XC_all  = np.concatenate((XC_all,X_noise));
        labels_ = np.concatenate((labels_,-1*np.ones((len(X_noise),),dtype=np.int)));
 
        #***********************************
        
        self.labels_groundtruth = labels_;
        self.XC                 = XC_all;
        
        self.N_Noise            = N_Noise;
        self.seed               = seed;
       
    # def __generateSamplePoints(self):
        
    #     params = self.parameters;
                
    #     x_ = np.random.uniform(0,params["x_scale"],params["N_SamplePoints"]);     
    #     y_ = params["y_scale_sigma"]*np.sin(3.14*x_/params["x_scale"]) +\
    #             np.random.normal(loc=0.0,scale=params["y_scale_noise"],size=params["N_SamplePoints"]);
    #     XC_Sample = (np.array([x_,y_])).T;
        
    #     self.XC_Sample  = XC_Sample;