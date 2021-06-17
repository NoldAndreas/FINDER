import numpy as np
import time
import pandas as pd
from sklearn.cluster import DBSCAN
from DbscanLoop import DbscanLoop
from sklearn.neighbors import NearestNeighbors
from ProgressBar import printProgressBar

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
        self.no_points_thresholds = points_per_dimension;
        self.algo                = algo;
        self.one_two_d           = "twoD";
        self.minmax_threshold    = [3,30];

    #**************************************************
    # fit
    #**************************************************

    def fit(self,XC,XC_params=[],**kwargs):
        
        print("Analysing "+str(len(XC))+" points");

        if(XC_params==[]):
            XC_params = XC;

        t_1 = time.time();
        #Step 1: Get min max of threshold and sigma
        if(self.one_two_d == "oneD"):
            params = self.__getParams_Sigmas(XC_params);
        if(self.one_two_d == "oneD_thresholds"):
            params = self.__getParams_Thresholds(XC_params);
        elif(self.one_two_d == "twoD"):
            params = self.__getParams_SigmasThresholds(XC_params);
        t_2 = time.time();


        #Step 2: Compute phase spaces
        phasespace    = self.__phaseSpace(XC,params);
        self.phasespace          = phasespace;
        t_3 = time.time();

        if(("skipSimilarityScore" in kwargs.keys()) and (kwargs['skipSimilarityScore']==True)):
            self.computationTimes    = {'Step1':t_2-t_1,'Step2':t_3-t_2};
            print("Skipping similarity score and computation of optimum")
            return False;

        #Step 3: Compute similarity score
        data          = self.__phaseSpacePostProcess(XC,phasespace);
        t_4 = time.time();


        #Step 3: Get parameterset
        labels,selected_parameters = self.__get_consensus_clustering(data,XC);

        print("Comp time Step 1 (set boundaries): "+str(np.round(t_2-t_1,2))+" seconds");
        print("Comp time Step 2 (clustering): "+str(np.round(t_3-t_2,2))+" seconds");
        print("Comp time Step 3 (postprocess, similarity scores): "+str(np.round(t_4-t_3,2))+" seconds");

        #Save data
        self.computationTimes    = {'Step1':t_2-t_1,'Step2':t_3-t_2,'Step3':t_4-t_3};
        self.data                = data;
        self.labels              = labels;
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

        for i, ps in PS.iterrows():
            no_clusters = np.max(ps["labels"]) + 1;

            centers_i = np.zeros((no_clusters,2));
            radii_i   = np.zeros((no_clusters,));

            #go through all clusters:
            for icl in np.arange(no_clusters):
                XCm = XC[(ps["labels"] == icl)];

                c                = np.median(XCm,axis=0);
                centers_i[icl,:] = c;
                radii_i[icl]     = np.max(np.linalg.norm(XCm - c,axis=1));

            centers.append(centers_i);
            radii.append(radii_i);

        return centers,radii

    #**************************************************
    #
    #**************************************************
    def __getParams_SigmasThresholds(self,XC):
        minmax_sigma = self.__determine_sigma_boundaries(XC);
        sigmas       = self.__getLogDistribution(minmax_sigma[0],minmax_sigma[1],self.no_points_sigma);

        thresholds       = self.__getLogDistribution(self.minmax_threshold[0],self.minmax_threshold[1],self.no_points_thresholds);
        thresholds       = np.unique(np.round(thresholds));

        s_all = [];
        t_all = [];
        for s in sigmas:
            for t in thresholds:
                s_all.append(s);
                t_all.append(t);

        params        = pd.DataFrame(data={"sigma":np.asarray(s_all),\
                                           "threshold":np.asarray(t_all)});
        return params


    def __getParams_Sigmas(self,XC):
        minmax_sigma =  self.__determine_sigma_boundaries(XC);
        sigmas        = self.__getLogDistribution(minmax_sigma[0],minmax_sigma[1],self.no_points_sigma);
        params        = pd.DataFrame(data={"sigma":sigmas,\
                                           "threshold":self.threshold*np.ones_like(sigmas)});
        return params

    def __getParams_Thresholds(self,XC):

        k         = 10;
        # initialize model
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(XC);
        dist_, ind = neigh.kneighbors(XC);

        # We have no use of indices here
        # dist is a 2 dimensional array of shape (10000, 9) in which each row is list of length 9. This row contain distances to all 9 nearest points. But we need distance to only 9th nearest point. So
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]

        #CD_sorted = np.sort(dist.squareform(dist.pdist(XC)),axis=1);
        sigma = np.quantile(nPt_distance,0.5);


        thresholds = self.__getLogDistribution(self.minmax_threshold[0],self.minmax_threshold[1],self.no_points_sigma);
        thresholds = np.unique(np.round(thresholds));

        params     = pd.DataFrame(data={"sigma":sigma*np.ones_like(thresholds),\
                                        "threshold":thresholds});
        return params
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
    def __phaseSpace(self,XC,params):

        labels_all = [];
        times      = [];
        
        t1_all = time.time()
        
        printProgressBar(0, len(params), prefix = 'Clustering progress:', suffix = 'Complete', length = 50);
        for index, param in params.iterrows():
            t1 = time.time()
            labels_ = self.ComputeClusters(param['sigma'],param['threshold'],XC);
            t2 = time.time()
            labels_all.append(labels_);
            times.append(t2-t1);
            printProgressBar(index + 1, len(params), prefix = 'Progress:', suffix = 'Complete', length = 50)
            #print("Computing time for sigma = "+str(np.round(param['sigma'],2))+" and minPts ="+ str(param['threshold'])+" : " + str(np.round(t2-t1,2)) );            

        print("Computing clusters : " + str(np.round(time.time()-t1_all,2)) );            
        ps = params;
        ps['labels'] = labels_all;
        ps['time']   = times;

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
        printProgressBar(0, len(PS), prefix = 'Postprocessing progress:', suffix = 'Complete', length = 50);
        for i, ps in PS.iterrows():
            for j in np.arange(i+1):
                if(not (i==j)):
                    score = self.__getSimilarityScore(i,j,PS,centers,radii);
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
            no_clusters[i]         = np.int(np.max(ps["labels"]) + 1);
            similarityScore[i]     = np.sum(similarityScoreMatrix[i,:]);
            times[i]               = ps["time"];

        PS["no_clusters"]     = no_clusters;
        PS["time"]            = times;
        PS["similarityScore"] = similarityScore;
        
        self.similarityScoreMatrix = similarityScoreMatrix;

        return PS


    #**************************************************
    # __findRelevantClusters
    #**************************************************
    def __get_consensus_clustering(self,PS,XC):

        similarity = np.asarray(PS["similarityScore"]);

        max_score  = np.max(similarity);

        idx        = np.argwhere(similarity == max_score)[-1][0]

        sigma_selected     = PS.loc[idx,'sigma'];
        threshold_selected = PS.loc[idx,"threshold"];


        labels = PS.loc[idx,'labels'];
        selected_parameters =  {"sigma":sigma_selected,
                                "threshold":threshold_selected};

        print("Selected threshold , sigma : "+str(threshold_selected)+" , " + str(sigma_selected));
        return labels,selected_parameters;

    #**************************************************
    # __getSimilarityScore
    #**************************************************
    def getSimilarityScoreDistribution(self,XC,i):
        
        PS = self.phasespace;
        print(PS.loc[i,:]);
        labels_1      = PS.loc[i,"labels"];
        centers,radii = self.__computeCenters_Radii(XC,PS);
        
        n1               = np.max(labels_1)+1;
        similarityScores = np.zeros((n1,),dtype=int)
        
        for j,ps in PS.iterrows():
            labels_2 = ps["labels"];
            
            n1                  = np.max(labels_1)+1;      
            n2                  = np.max(labels_2)+1;
            
            radii_1   = radii[i];
            radii_2   = radii[j];
            centers_1 = centers[i];
            centers_2 = centers[j]; 
            
            for i1 in np.arange(n1):
                for i2 in np.arange(n2):
                    
                    similarityScores[i1] += self.__getSimilarityClusters_withPrecheck(labels_1,labels_2,i1,i2,centers_1,centers_2,radii_1,radii_2);
        return similarityScores;
    
    def __getSimilarityScore(self,i,j,PS,centers,radii):

        labels_1  = PS.loc[i,"labels"];
        labels_2  = PS.loc[j,"labels"];
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

#                if(self.__checkNoOverlapClusters(centers_1[i1,:],centers_2[i2,:],radii_1[i1],radii_2[i2])):
#                    similarityMatrix[i1,i2] = 0;
#                    continue;
#
#                similarity = self.__getSimilarityClusters(labels_1,labels_2,i1,i2);
                
                similarity = self.__getSimilarityClusters_withPrecheck(labels_1,labels_2,i1,i2,centers_1,centers_2,radii_1,radii_2);

                if(similarity):
                    similarityMatrix[i1,:]  = 0;
                    similarityMatrix[:,i2]  = 0;
                    similarityMatrix[i1,i2] = 1;
                    break;
                else:
                    similarityMatrix[i1,i2] = 0;

        return np.sum(similarityMatrix);


    def __getSimilarityClusters_withPrecheck(self,labels_1,labels_2,i1,i2,centers_1,centers_2,radii_1,radii_2):
        if(self.__checkNoOverlapClusters(centers_1[i1,:],centers_2[i2,:],radii_1[i1],radii_2[i2])):
            return 0
        else:                    
            return self.__getSimilarityClusters(labels_1,labels_2,i1,i2);

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
