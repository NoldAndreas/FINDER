import numpy as np
import json
import os.path
import pickle

from Finder_1d import Finder_1d

from FigX4_Explore2DOptimizer_withReference_Streamlined_Functions import GetDensity
from FigX4_Explore2DOptimizer_withReference_Streamlined_Functions import GetOverlay
from FigX4_Explore2DOptimizer_withReference_Streamlined_Functions import LoadPoints
from FigX4_Explore2DOptimizer_withReference_Streamlined_Functions import FilterPoints

class ClusterBasing:

    def __init__(self,basefolder,parameterfile):

        #Load Parameter file
        #parameterfile = 'MikeData/Analysis_dataWindow_1/dataWindow_1_parameters';
        with open(basefolder+parameterfile+'.json') as f:
            parameters = json.load(f);

        if(not ('datascale' in parameters.keys())):
            parameters['datascale'] = 1;

        parameters['outputfolder']   = parameters['mainfolder'] + 'Analysis_'+parameters['analysis_name']+'/';
        parameters['save_name']      = parameters['outputfolder']+parameters['analysis_name'];
        parameterfile                = basefolder+parameters['save_name']+'_parameters.json';

        self.basefolder              = basefolder;
        self.parameters              = parameters;
        self.save_name               = basefolder + parameters['save_name'];

        #Load Points
        self.__loadPoints(basefolder,parameters);

    def __loadPoints(self,basefolder,parameters):

        if(os.path.isfile(basefolder+parameters['save_name']+"_filtered_incell.txt")):
            XC_incell  = LoadPoints(basefolder+parameters['save_name']+"_filtered_incell.txt",datascale=parameters['datascale']);
            XC_outcell = LoadPoints(basefolder+parameters['save_name']+"_filtered_outcell.txt",datascale=parameters['datascale']);
        else:
            XC_incell  = LoadPoints(basefolder+parameters['mainfolder']+parameters['image_filename']+'_incell.txt',datascale=parameters['datascale']);
            XC_outcell = LoadPoints(basefolder+parameters['mainfolder']+parameters['image_filename']+'_outcell.txt',datascale=parameters['datascale']);

            XC_incell   = FilterPoints(XC_incell,parameters['incell_window']);
            XC_outcell  = FilterPoints(XC_outcell,parameters['outcell_window']);

            XC_outcell_overlay = GetOverlay(XC_incell,XC_outcell);

            np.savetxt(basefolder+parameters['save_name']+"_filtered_incell.txt",XC_incell,fmt="%f\t%f");
            np.savetxt(basefolder+parameters['save_name']+"_filtered_outcell.txt",XC_outcell,fmt="%f\t%f");

        self.XC_incell = XC_incell;
        self.XC_outcell = XC_outcell;

    def GetClusterings(self):

        parameters = self.parameters;
        filename   = self.basefolder+parameters['outputfolder']+"results_"+parameters['analysis_name'];

        #******************************************************************************************
        if(os.path.exists(filename+'_incell.pickle')):
            with open(filename+'_incell.pickle', 'rb') as fr:
                FD_load = pickle.load(fr);
            FD     = FD_load['FD'];

            print("Loaded Clustering results from "+filename+'_incell.pickle');
        else:
            FD      = Finder_1d(algo=parameters['algo']);
            labels  = FD.fit(XC_incell);

            with open(filename+'_incell.pickle','wb') as handle:
                pickle.dump({'FD':FD}, handle,protocol=pickle.HIGHEST_PROTOCOL)
            print("Computed and saved Clustering results in "+filename+'_incell.pickle');

        #******************************************************************************************
        if(os.path.exists(filename+'_outcell.pickle')):
            with open(filename+'_outcell.pickle', 'rb') as fr:
                FD_load = pickle.load(fr);
            FD_ref  = FD_load['FD_ref'];

            print("Loaded Clustering results from "+filename+'_outcell.pickle');
        else:
            FD_ref      = Finder_1d(algo=parameters['algo']);
            labels_ref  = FD.fit(XC_incell);

            with open(filename+'_outcell.pickle','wb') as handle:
                pickle.dump({'FD':FD_ref}, handle,protocol=pickle.HIGHEST_PROTOCOL)
            print("Computed and saved Clustering results in "+filename+'_outcell.pickle');

        return False;

    def GetReferenceClustering(self):
        return False;

    def PostProcess(self):
        return False;
