import numpy as np
import json
import os.path
import pickle

from Finder_1d import Finder_1d

from SimilarityScore import getSimilarityScore,getClusterSizesAll,getSimilarityScore_ij

from FigX4_Explore2DOptimizer_withReference_Streamlined_Functions import GetDensity,GetOverlay
from FigX4_Explore2DOptimizer_withReference_Streamlined_Functions import LoadPoints,FilterPoints
from FigX4_Explore2DOptimizer_withReference_Streamlined_Functions import DefineCleanedLabels_GeneralLimit,GetLineOfOptima

class ClusterBasing:

    def __init__(self,basefolder,algo='DbscanLoop'):

        #Load Parameter file
        parameters= {'algo':algo};

        if not basefolder.endswith(os.path.sep):
            basefolder += os.path.sep

        with open(basefolder+'parameters_clusterBasing.json', 'w') as fp:
            json.dump(parameters,fp,indent=4);

        if(os.path.isfile(basefolder+"X_incell_window.txt") and \
            os.path.isfile(basefolder+"X_outcell_window.txt")):
            self.XC_incell  = LoadPoints(basefolder+"X_incell_window.txt");
            self.XC_outcell = LoadPoints(basefolder+"X_outcell_window.txt");
        else:
            print("X_incell_window or X_outcell_window in folder "+basefolder+" not found!");

        self.basefolder              = basefolder;
        self.parameters              = parameters;
        self.save_name               = basefolder + 'analysis';

    def GetClusterings_InOutCell(self):

        parameters = self.parameters;
        filename   = self.save_name+"clustering";

        #******************************************************************************************
        # Load or Compute Clustering within cell
        #******************************************************************************************
        if(os.path.exists(filename+'_incell.pickle')):
            with open(filename+'_incell.pickle', 'rb') as fr:
                FD_load = pickle.load(fr);
            FD     = FD_load['FD'];

            print("Loaded Clustering results from "+filename+'_incell.pickle');
        else:
            FD      = Finder_1d(algo=parameters['algo']);
            labels  = FD.fit(self.XC_incell,skipSimilarityScore=True);

            with open(filename+'_incell.pickle','wb') as handle:
                pickle.dump({'FD':FD}, handle,protocol=pickle.HIGHEST_PROTOCOL)
            print("Computed and saved Clustering results in "+filename+'_incell.pickle');

        #******************************************************************************************
        # Load or Compute Clustering outside cell
        #******************************************************************************************
        if(os.path.exists(filename+'_outcell.pickle')):
            with open(filename+'_outcell.pickle', 'rb') as fr:
                FD_load = pickle.load(fr);
            FD_ref  = FD_load['FD'];

            print("Loaded Clustering results from "+filename+'_outcell.pickle');
        else:
            FD_ref      = Finder_1d(algo=parameters['algo']);
            labels_ref  = FD_ref.fit(self.XC_outcell,self.XC_incell,skipSimilarityScore=True);

            with open(filename+'_outcell.pickle','wb') as handle:
                pickle.dump({'FD':FD_ref}, handle,protocol=pickle.HIGHEST_PROTOCOL)
            print("Computed and saved Clustering results in "+filename+'_outcell.pickle');

        #******************************************************************************************
        # Assemble data
        #******************************************************************************************
        phasespace_all                    = FD.phasespace;
        phasespace_all['labels_ref']      = FD_ref.phasespace['labels']
        phasespace_all['no_clusters_ref'] = FD_ref.phasespace['no_clusters'];
        phasespace_all['time_ref']        = FD_ref.phasespace['time'];
        self.phasespace_all               = phasespace_all;

        df_clusterSizes     = FD.clusterInfo;#GetClusterSizesAll(FD);
        df_clusterSizes_ref = FD_ref.clusterInfo;#'GetClusterSizesAll(FD_ref);

        df_clusterSizes['type']     = 'incell';
        df_clusterSizes_ref['type'] = 'outcell';
        self.df_clusterSizes_all         = df_clusterSizes.append(df_clusterSizes_ref, ignore_index=True);

    def GetReferenceClustering(self,bestRequiredRate=1.0,computeSimilarityScores=False):

        #*********************************************
        # Get limit and filter by cluster size
        #*********************************************
        phasespace_all_aboveT = DefineCleanedLabels_GeneralLimit(self.df_clusterSizes_all,self.phasespace_all,criterion='clusterSize',bestRequiredRate=bestRequiredRate);
        clusterInfo_aboveT    = getClusterSizesAll(self.XC_incell,phasespace_all_aboveT);

        if(computeSimilarityScores==True):
            cli_similarityScore,similarityScore      = getSimilarityScore(self.XC_incell,phasespace_all_aboveT,clusterInfo_aboveT);
            phasespace_all_aboveT['similarityScore'] = similarityScore;
            clusterInfo_aboveT['similarityScore']    = cli_similarityScore;

        self.phasespace_all_aboveT = phasespace_all_aboveT;
        self.clusterInfo_aboveT    = clusterInfo_aboveT;
        self.df_opt_th_aboveT_ncl  = GetLineOfOptima(self.phasespace_all_aboveT[['sigma', 'threshold','no_clusters']],'threshold','no_clusters');


    def GetClustering(self,criterion='percent_locsIncluded'): #'no_clusters'

        df_opt_th_aboveT_ncl = GetLineOfOptima(self.phasespace_all_aboveT[['sigma', 'threshold','no_clusters','percent_locsIncluded']],'threshold','no_clusters');
        i_choose             = df_opt_th_aboveT_ncl.loc[df_opt_th_aboveT_ncl[criterion].argmax(),'idx'];
        #i_choose = np.argmax(self.phasespace_all_aboveT[criterion]);
        #i_check = 56;
        v = [];
        for i in np.arange(len(self.phasespace_all_aboveT)):
            s1= getSimilarityScore_ij(i_choose,i,self.phasespace_all_aboveT,self.clusterInfo_aboveT);
        #    print(s1)

            if(type(s1)!= bool):
                v.append(np.sum(s1[0]));
            else:
                v.append(0);
         #       print(np.sum(s1[0]),np.sum(s1[1]));
        self.phasespace_all_aboveT['similarityScoreChosen'] = v;
        print(self.phasespace_all_aboveT.loc[i_choose,:]);

        return self.phasespace_all_aboveT.loc[i_choose,:];
