#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:51:43 2020

@author: andreas
"""

import numpy as np
from sklearn.cluster import OPTICS
from scipy.spatial.distance import euclidean
from Clustering_CAML import Clustering_CAML
from Finder_1d import Finder_1d
from sklearn.cluster import DBSCAN
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Clustering class

class Clustering:
    
    def __init__(self,G,basefolder):
        print(basefolder);
        self.Geometry   = G;
        self.basefolder = basefolder;
        self.min_overlap_per_ref = 0.3;
        
    def fit(self,algo,params=[]):
           
        self.algo = algo;
        XC = self.Geometry.XC;
        
        result_ = [];
        
        t_start = time.time()   
        
        if(algo == "OPTICS"):
            params_     = params['OPTICS'];
            threshold   = params_["min_samples"];
            xi          = params_["xi"];
            eps_max     = params_["max_eps"];
            
            clustering_optics = OPTICS(min_samples=threshold, xi=xi,max_eps=eps_max,metric=euclidean,cluster_method='xi').fit(XC); # min_cluster_size=.05
            labels = clustering_optics.labels_;
        if('CAML' in algo):
            labels = Clustering_CAML(algo,XC,datafolder=self.basefolder);       
        elif(algo == 'FINDER_1D_loop'):                
            FD      = Finder_1d(algo='DbscanLoop');
            labels  = FD.fit(XC);                
            result_ = FD.selected_parameters;
        elif(algo == 'FINDER_1D'):                
            FD      = Finder_1d(algo='dbscan');
            labels  = FD.fit(XC);                
            result_ = FD.selected_parameters;
        elif(algo == 'dbscan'):
            params_     = params['dbscan'];
            eps         = params_['eps']; 
            min_samples = params_['min_samples'];
            DB          = DBSCAN(eps=eps,min_samples=min_samples).fit(XC);
            labels      = DB.labels_;
        
        #clustering_optics = OPTICS(min_samples=threshold, xi=sigma,max_eps=0.3,metric=euclidean,cluster_method='xi').fit(XC) # min_cluster_size=.05
        
        self.computationTime = time.time() - t_start;
        self.labels = labels;
        return result_;
        
    def Evaluate(self):
                
        self.__labelEvaluation();    
        self.__clusterEvaluation();           
        self.__locEvaluation();
        self.__subClustersPerRefCluster();
        
    def __subClustersPerRefCluster(self):
        #Return an array. Entry i is how many subclusters reference cluster
        # i is devided into
        
        labels_groundTruth  = self.Geometry.labels_groundtruth;
        labels              = self.labels;
        
        n_cl                = np.max(labels_groundTruth);
        
        number_of_subclusters = np.zeros((n_cl+1,));
        
        for idx_ref in np.unique(labels_groundTruth): 
            if(idx_ref == -1):
                continue;
            #Get the points 
            mark = ((labels_groundTruth == idx_ref)*(labels != -1));
            
            number_of_subclusters[idx_ref] = len(np.unique(labels[mark]));
        self.number_of_subclusters = number_of_subclusters;
            
        
    def __labelEvaluation(self):
        
        labels              = self.labels;
        labels_groundTruth  = self.Geometry.labels_groundtruth;
                 
        no_clusters         = np.max(labels) +1;
        no_clusters_gT      = np.max(labels_groundTruth) +1;
        

        min_overlap_per_ref = self.min_overlap_per_ref;
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
                    
                    if(len(cl_and_clRef)/len(cl) > coverage):
                        idx_chosen = ii;
                        coverage   = len(cl_and_clRef)/len(cl);
            
            clusters_ref =  [clusters_ref[i] for i in np.arange(len(clusters_ref)) if (i!=idx_chosen)]
            clusters_correct[i] = idx_chosen;                    
                    
        self.clusters_correct = clusters_correct;    
        return clusters_correct        
        
    def __clusterEvaluation(self):
        
        clusters_correct = self.clusters_correct;
        
        n_groundtruth = np.max((self.Geometry.labels_groundtruth)) + 1;
        n_identified = np.max((self.labels)) + 1;
        
        true_positives  = np.sum(clusters_correct != -1);
        false_positives = np.sum(clusters_correct == -1);
        false_negatives = n_groundtruth - true_positives;
        
        all_groundtruth = np.arange(n_groundtruth);
        all_identified  = np.arange(n_identified);
                
        false_negative_clusters = np.asarray([a for a in all_groundtruth if (not (a in list(clusters_correct)))]);
        false_positive_clusters = np.asarray([i for i in all_identified  if (clusters_correct[i] == -1)]);
        true_positive_clusters  = np.asarray([i for i in all_identified  if (clusters_correct[i] != -1)]);
        
        cluster_evaluation = {"true_positives":true_positives,
                              "false_positives":false_positives,
                              "false_negatives":false_negatives,
                              "n_groundtruth":n_groundtruth,
                              "false_negative_clusters":false_negative_clusters,
                              "false_positive_clusters":false_positive_clusters,
                              "true_positive_clusters":true_positive_clusters};
        #COMPUTE
        self.cluster_evaluation = cluster_evaluation;
        return cluster_evaluation
        
    def __locEvaluation(self):
          
        labels             = self.labels;
        labels_groundTruth = self.Geometry.labels_groundtruth;
        
        false_positives    = np.sum((labels != -1) * (labels_groundTruth == -1));
        false_negatives    = np.sum((labels == -1) * (labels_groundTruth != -1));
        true_positives     = np.sum((labels != -1) * (labels_groundTruth != -1)); 
        
        loc_evaluation = {"true_positives":true_positives,
                          "false_positives":false_positives,
                          "false_negatives":false_negatives};
                          
        self.loc_evaluation = loc_evaluation;
        return loc_evaluation;
    
    def PlotScatter(self,filename):
 
        labels = self.labels;
        XC     = self.Geometry.XC;
        
        # Get correctly detected:
        correct_detected = np.zeros_like(labels);
        for i,cl_ in enumerate(self.clusters_correct):
            if(cl_ != -1): #if not correctly detected
                correct_detected[labels==i] = 1;
        
        fig,ax = plt.subplots();
        mark = (labels==-1);
        sns.scatterplot(x=XC[mark,0],y=XC[mark,1],color='grey',alpha=0.2);
        mark = (labels>=0);
        sns.scatterplot(x=XC[mark,0],y=XC[mark,1],hue=labels[mark],palette='bright',
                        size=0.2,style=-1*correct_detected[mark],legend=False);
        ax.set_aspect('equal');
        
        
        x_0 = 0;
        y_0 = np.min(XC[:,1]) - 50;
        ax.plot([x_0,x_0+100],[y_0,y_0],'k')
        ax.annotate('$100nm$',(x_0+50,y_0+10),fontsize='large',ha='center');         
        ax.set_aspect(1);
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.axis('off');

        plt.savefig(filename)
   
    
        
