import numpy as np
import time
import pandas as pd
from sklearn.cluster import DBSCAN 
from DbscanLoop import DbscanLoop
from sklearn.neighbors import NearestNeighbors

#**************************************************
# FINDER
#
# Parameters:
# threshold (default = 10)
# points_per_dimension (default = 15)
# algo (default = DbscanLoop) 
#
#**************************************************

class Finder_1d:
    
    #**************************************************
    #**************************************************
    # Public functions
    #**************************************************
    #**************************************************

    #**************************************************
    # Initialization
    #**************************************************    
    def __init__(self,threshold=10,points_per_dimension=15,algo="DbscanLoop"):
        self.threshold           = np.int(threshold);
        self.no_points_sigma     = points_per_dimension;
        self.algo                = algo;

    #**************************************************
    # fit
    #**************************************************
            
    def fit(self,XC):
        
        
        #Step 1: Get min max of threshold and sigma
        minmax_sigma =  self.__determine_sigma_boundaries(XC);
        
        #Step 2: Compute phase spaces
        phasespace = self.__phaseSpaceLogDomain(XC,minmax_sigma=minmax_sigma);
        
        #Step 3: Compute similarity score
        data          = self.__phaseSpacePostProcess(XC,phasespace);    
        
        #Step 3: Get parameterset
        labels,selected_parameters = self.__get_consensus_clustering(data,XC);        
        
        #Save data
        self.data                = data;
        self.labels              = labels;
        self.phasespace          = phasespace;
        self.selected_parameters = selected_parameters;
        
        return labels;
            
    #**************************************************
    # Find Clusters
    #**************************************************
    def GetSimilarity(self,labels_1,labels_2):
        sim_ = self.__getSimilarity(labels_1,labels_2);
        if(sim_):
            print("similar clusters");
        else:
            print("not similar clusters");
        return sim_;
    
    #**************************************************
    #**************************************************
    # Private functions
    #**************************************************
    #**************************************************
    
    def __computeCenters_Radii(self,XC,PS):
        centers = [];
        radii   = [];        
        for i,ps in enumerate(PS): 
            no_clusters = np.max(ps["labels"]) + 1;
                        
            centers_i = np.zeros((no_clusters,2));
            radii_i   = np.zeros((no_clusters,));
            
            #go through all clusters:
            for icl in np.arange(no_clusters):
                XCm = XC[(ps["labels"] == icl)];
                
                c                = np.median(XCm,axis=0);
                centers_i[icl,:] = c;
                radii_i[icl]     = np.max(np.linalg.norm(XCm - c,axis=1));
                #np.linalg.norm(np.max(XCm,axis=0) - np.min(XCm,axis=0));
        
            centers.append(centers_i);
            radii.append(radii_i);
            
        return centers,radii
    #**************************************************    
    # Check if number of points is too low and adjust 
    #  minmax_threshold
    #**************************************************                            
    def __determine_sigma_boundaries(self,XC):
        
        k         = self.threshold+1;    
        # initialize model
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(XC);
        dist_, ind = neigh.kneighbors(XC);
    
        # We have no use of indices here
        # dist is a 2 dimensional array of shape (10000, 9) in which each row is list of length 9. This row contain distances to all 9 nearest points. But we need distance to only 9th nearest point. So
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]
        
        #CD_sorted = np.sort(dist.squareform(dist.pdist(XC)),axis=1);      
        sigma_min = np.quantile(nPt_distance,0.1);
        sigma_max = np.quantile(nPt_distance,0.9);    
                                             
        minmax_sigma = [sigma_min,sigma_max];
                
        print( "Boundaries for sigma    : " + str(minmax_sigma[0]) + " , " + str(minmax_sigma[1]));
        
        return minmax_sigma
    
    #**************************************************
    # __phaseSpaceLogDomain
    #**************************************************
    def __phaseSpaceLogDomain(self,XC,minmax_sigma):

        print("Computing clustering results within sigma boundaries..");
        sigmas        = self.__getLogDistribution(minmax_sigma[0],minmax_sigma[1],self.no_points_sigma);                
        ps            = self.__phaseSpace(sigmas,XC);                

        return ps;
    
    #**************************************************
    # __getLogDistribution
    #**************************************************
    def __getLogDistribution(self,min_x,max_x,n):
        min_log = np.log(min_x);
        max_log = np.log(max_x);    
        
        log_vec = np.linspace(min_log,max_log,n);
        
        vec     = np.exp(log_vec);
        vec     = np.unique(vec);        
        
        return vec;
    
    #**************************************************
    # __phaseSpace
    #**************************************************
    def __phaseSpace(self,sigmas,XC=[]):
        
        ps = [];
        for sigma in sigmas:            
            t1 = time.time()   
            labels_ = self.ComputeClusters(sigma,self.threshold,XC);
            t2 = time.time()                                
            datapoint = {"labels":labels_,"sigma":sigma,
                         "threshold":self.threshold,"time":t2-t1};
            ps.append(datapoint)
                    
        return ps

    #**************************************************
    # ComputeClusters
    #**************************************************
    def ComputeClusters(self,sigma,threshold,XC):

        if((self.algo == "dbscan")):
            DB          = DBSCAN(eps=sigma,min_samples=threshold).fit(XC);
            labels_     = DB.labels_;        
        elif((self.algo == "DbscanLoop")):
            DBL     = DbscanLoop(eps=sigma,min_samples=threshold).fit(XC);
            labels_ = DBL.labels_;       
        else:
            self.__print("ALGORITHM NOT RECOGNIZED !!");   
        return labels_
    
                        

    #**************************************************
    # __phaseSpacePostProcess
    #  -- Computes similarity scores
    #**************************************************            
    def __phaseSpacePostProcess(self,XC,PS):
                                        
        print("Postprocessing..") 
        
        n = len(PS);
    
        sigmas                   = np.zeros(shape=(n,))
        thresholds               = np.zeros(shape=(n,),dtype=np.int)         
        no_clusters              = np.zeros(shape=(n,),dtype=np.int)                     
        no_locs                  = np.zeros(shape=(n,),dtype=np.int) 
        times                    = np.zeros(shape=(n,))
        similarityScore          = np.zeros(shape=(n,))  
        similarityScoreMatrix    = np.zeros(shape=(n,n))  
        
        #***************************************
        # Preprocess: get centers and radii
        #***************************************
        centers,radii = self.__computeCenters_Radii(XC,PS);

        #***************************
        # Compute similarity scores
        #***************************        
        ###
        t1 = time.time();
        for i in np.arange(n):
            for j in np.arange(i+1):            
                if(not (i==j)):
                    score = self.__getSimilarityScore(i,j,PS,centers,radii);
                    similarityScoreMatrix[j,i] = score;
                    similarityScoreMatrix[i,j] = score;
                else:
                    similarityScoreMatrix[i,j] = np.max(PS[i]["labels"]) + 1;
        print("Computing similarity scores: "+str(time.time()-t1)+" seconds");
        
        #***************************
        # Collect data
        #***************************        
        for i,ps in enumerate(PS):  
            thresholds[i]          = ps["threshold"];
            sigmas[i]              = ps["sigma"];  
            labels_                = ps["labels"];
            no_clusters[i]         = np.int(np.max(labels_) + 1);
            similarityScore[i]     = np.sum(similarityScoreMatrix[i,:]);
            times[i]               = ps["time"];
        
        data = np.asarray([no_clusters,no_locs,sigmas,thresholds,times,similarityScore]);
        cols_ = ["no_clusters","no_locs","sigmas","thresholds","time","similarityScore"];
        df = pd.DataFrame(data.T,columns=cols_);
                
        return df    
    
    
    #**************************************************
    # __findRelevantClusters
    #**************************************************                
    def __get_consensus_clustering(self,phasespace,XC):
               
        similarity        = np.asarray(phasespace["similarityScore"]);
        
        max_score = np.max(similarity);
        mark       = (similarity ==max_score);
        
        sigma_selected     = (np.asarray(phasespace["sigmas"]))[mark];    
        threshold_selected = (np.asarray(phasespace["thresholds"]))[mark];
        
        if(np.sum(mark)>1):
            sigma_selected = sigma_selected[-1];
            threshold_selected = threshold_selected[-1]
        else:
            sigma_selected = sigma_selected[0];
            threshold_selected = threshold_selected[0]


        labels = self.ComputeClusters(sigma_selected,threshold_selected,XC);
        selected_parameters =  {"sigma":sigma_selected,
                                "threshold":threshold_selected}; 

        print("Selected sigma for threshold = "+str(threshold_selected)+" : " + str(sigma_selected));
        return labels,selected_parameters;
    
    #**************************************************
    # __getSimilarityScore
    #************************************************** 
    def __getSimilarityScore(self,i,j,PS,centers,radii):
        
        labels_1  = PS[i]["labels"];
        labels_2  = PS[j]["labels"];
        radii_1   = radii[i];
        radii_2   = radii[j];
        centers_1 = centers[i];
        centers_2 = centers[j];        
        count     = 0;
        
        #return zero score if no clusters selected or 
        # if one cluster is selcted which covers most points
        if((np.max(labels_1) == -1) or (np.max(labels_2) == -1)):
            return count;
        elif((np.max(labels_1) == 0) and (np.sum(labels_1 == 0)/len(labels_1) > 0.5 )):  
            return count;
        elif((np.max(labels_2) == 0) and (np.sum(labels_2 == 0)/len(labels_2) > 0.5 )):  
            return count;
        
        #******************************        
        n1                  = np.max(labels_1)+1;
        n2                  = np.max(labels_2)+1;
        similarityMatrix    = np.zeros((n1,n2));
        similarityMatrix[:] = np.nan;
        #******************************        
        
        for i1 in np.arange(n1):
            for i2 in np.arange(n2):
                
                if(similarityMatrix[i1,i2]==0):
                    continue;
                
                if(self.__checkNoOverlapClusters(centers_1[i1,:],centers_2[i2,:],radii_1[i1],radii_2[i2])):
                    similarityMatrix[i1,i2] = 0;
                    continue;
                    
                similarity = self.__getSimilarityClusters(labels_1,labels_2,i1,i2);
                
                if(similarity):
                    similarityMatrix[i1,:]  = 0;
                    similarityMatrix[:,i2]  = 0;                    
                    similarityMatrix[i1,i2] = 1;
                    break;
                else:
                    similarityMatrix[i1,i2] = 0;
        
        return np.sum(similarityMatrix);
        
                
    def __checkNoOverlapClusters(self,c1,c2,r1,r2):
        return (np.linalg.norm(c2-c1) > r1 + r2);
        
    def __getSimilarityClusters(self,labels_1,labels_2,i1,i2):
        no_locs_1       = np.sum(labels_1 == i1);
        no_locs_2       = np.sum(labels_2 == i2);
        no_locs_overlap = np.sum((labels_1 == i1)*(labels_2 == i2));
                
        if((no_locs_overlap/no_locs_1 > 0.5) and (no_locs_overlap/no_locs_2 > 0.5)):
            return True;
        else:
            return False;
        
   