# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from ProgressBar import printProgressBar

def __checkNoOverlapClusters(c1,c2,r1,r2):
        return (np.linalg.norm(c2-c1) > r1 + r2);
    

def __getSimilarityClusters(labels_1,labels_2,i1,i2):
    no_locs_1       = np.sum(labels_1 == i1);
    no_locs_2       = np.sum(labels_2 == i2);
    no_locs_overlap = np.sum((labels_1 == i1)*(labels_2 == i2));

    if((no_locs_overlap/no_locs_1 > 0.5) and (no_locs_overlap/no_locs_2 > 0.5)):
        return True;
    else:
        return False;
    
    
def __computeCenters_Radii(XC,PS):
    centers = [];
    radii   = [];

    for i, ps in PS.iterrows():

        no_clusters = np.max(ps["labels"]) + 1;

        centers_i = np.zeros((no_clusters,2));
        radii_i   = np.zeros((no_clusters,));
        
        #go through all clusters:
        for icl in np.unique(ps["labels"]):
            if(icl == -1):
                continue;
            XCm = XC[(ps["labels"] == icl)];

            c                = np.median(XCm,axis=0);
            centers_i[icl,:] = c;
            radii_i[icl]     = np.max(np.linalg.norm(XCm - c,axis=1));

        centers.append(centers_i);
        radii.append(radii_i);

    return centers,radii        
    
# def __computeCenters_Radii(XC,PS):
#     centers = [];
#     radii   = [];

#     for i, ps in PS.iterrows():
#         no_clusters = np.max(ps["labels"]) + 1;

#         centers_i = np.zeros((no_clusters,2));
#         radii_i   = np.zeros((no_clusters,));

        
#         #go through all clusters:
#         for icl in np.arange(no_clusters):
#             XCm = XC[(ps["labels"] == icl)];

#             c                = np.median(XCm,axis=0);
#             centers_i[icl,:] = c;
#             radii_i[icl]     = np.max(np.linalg.norm(XCm - c,axis=1));

#         centers.append(centers_i);
#         radii.append(radii_i);

#     return centers,radii    
    
def __getSimilarityClusters_withPrecheck(labels_1,labels_2,i1,i2,centers_1,centers_2,radii_1,radii_2):
    if(__checkNoOverlapClusters(centers_1[i1,:],centers_2[i2,:],radii_1[i1],radii_2[i2])):
        return 0
    else:                    
        return __getSimilarityClusters(labels_1,labels_2,i1,i2);


def getSimilarityScore_ij(i,j,PS,centers,radii):

    labels_1  = PS.loc[i,"labels"];
    labels_2  = PS.loc[j,"labels"];
    radii_1   = radii[i];
    radii_2   = radii[j];
    centers_1 = centers[i];
    centers_2 = centers[j];

    #return zero score if no clusters selected or
    # if one cluster is selcted which covers most points
    if((np.max(labels_1) == -1) or (np.max(labels_2) == -1)):
        return False;
    elif((np.max(labels_1) == 0) and (np.sum(labels_1 == 0)/len(labels_1) > 0.5 )):
        return False;
    elif((np.max(labels_2) == 0) and (np.sum(labels_2 == 0)/len(labels_2) > 0.5 )):
        return False;

    #******************************
    n1                  = np.max(labels_1)+1;
    n2                  = np.max(labels_2)+1;
    similarityMatrix    = np.zeros((n1,n2),dtype=int);
    similarityMatrix[:] = -1;
    #******************************
   
#    for i1 in np.arange(n1):
#        for i2 in np.arange(n2):
 
    for i1 in np.unique(labels_1):
        if(i1 == -1):
            continue;
        for i2 in np.unique(labels_2):
            if(i2 == -1):
                continue;            

            if(similarityMatrix[i1,i2]==0):
                continue;
            
            similarity = __getSimilarityClusters_withPrecheck(labels_1,labels_2,i1,i2,centers_1,centers_2,radii_1,radii_2);

            if(similarity):
                similarityMatrix[i1,:]  = 0;
                similarityMatrix[:,i2]  = 0;
                similarityMatrix[i1,i2] = 1;
                break;
            else:
                similarityMatrix[i1,i2] = 0;

    similarityMatrix[similarityMatrix==-1] = 0;

    s_i = np.sum(similarityMatrix,axis=1);
    s_j = np.sum(similarityMatrix,axis=0)
    return [s_i,s_j];


def getSimilarityScore(XC,PS,clusterInfo):
    
    t1 = time.time();
 
    #***************************************
    # Preprocess: get centers and radii
    #***************************************
    cli_index           = clusterInfo['index'];
    cli_similarityScore = np.zeros([len(cli_index),],dtype=int);
    n = len(PS);
    similarityScoreMatrix    = np.zeros(shape=(n,n));
    similarityScore          = np.zeros(shape=(n,))
 
    
    centers,radii = __computeCenters_Radii(XC,PS);
    
    printProgressBar(0, len(PS), prefix = 'Postprocessing progress:', suffix = 'Complete', length = 50);
    for i, ps in PS.iterrows():
        mark_i = (cli_index == i);

        for j in np.arange(i+1):                  
            mark_j = (cli_index == j);
            
            if(not (i==j)):

                s_ij = getSimilarityScore_ij(i,j,PS,centers,radii);
                
                if((type(s_ij)==bool) and (s_ij==False)):
                    continue;

                score = np.sum(s_ij[0]);

                cli_similarityScore[mark_i] += s_ij[0];
                cli_similarityScore[mark_j] += s_ij[1];
                    
                similarityScoreMatrix[j,i] = score; #/Normalize here?  eg  /np.int(np.max(PS.loc[j,"labels"]) + 1)
                similarityScoreMatrix[i,j] = score; #/Normalize here?  eg  /np.int(np.max(PS.loc[i,"labels"]) + 1)
            else:
                similarityScoreMatrix[i,j] = np.max(ps["labels"]) + 1;
        printProgressBar(i + 1, len(PS), prefix = 'Progress:', suffix = 'Complete', length = 50)
    print("Computing similarity scores: "+str(np.round(time.time()-t1,2))+" seconds");

    #***************************
    # Collect data
    #***************************

    for i, ps in PS.iterrows():       
        similarityScore[i]     = np.sum(similarityScoreMatrix[i,:]);
        
    
    return cli_similarityScore,similarityScore 



def getClusterSizesAll(PS):

      cl_sizes   = np.array([],dtype=int);
      thresholds = np.array([],dtype=int)
      sigmas     = np.array([]);
      labels     = np.array([],dtype=int);
      index      = np.array([],dtype=int);        
  
      for idx, df1_row in PS.iterrows():        
          
          labels_i = df1_row['labels'];
#          l_ = (np.unique(labels_i));
#          l_ = l_[l_>=0]         
          l_ = np.arange(np.max(labels_i)+1);
          
          if(l_.shape[0] == 0):
              continue;

          labels     = np.concatenate((labels,l_));        
          cl_sizes   = np.concatenate((cl_sizes,np.asarray([np.sum(labels_i==l) for l in l_])))            
          thresholds = np.concatenate((thresholds,df1_row['threshold']*np.ones_like(l_)));
          sigmas     = np.concatenate((sigmas,df1_row['sigma']*np.ones([len(l_),])));
          index      = np.concatenate((index,idx*np.ones_like(l_)));
  
      df_clusterSizes = pd.DataFrame();
      df_clusterSizes['labels']      = labels;
      df_clusterSizes['clusterSize'] = cl_sizes;
      df_clusterSizes['threshold']   = thresholds;
      df_clusterSizes['sigma']       = sigmas;  
      df_clusterSizes['index']       = index;          
      
      return df_clusterSizes;
