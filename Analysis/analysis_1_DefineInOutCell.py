"""splits localizations into inside and outside cell and clusters subset"""
from cell_segmentation import CellSegmentation

CellSegmentation("/Users/andreas/Documents/PostDoc/Clustering/\
NoiseRecognizer_WorkingVersion/AnalysisDataOrganized/TTX_control_new/").segmentation()


#from ClustersInOutCell import ClustersInOutCell
#filename      = sys.argv[1];
#OUTPUTFOLDER  = sys.argv[2];

#BASEFOLDER = "/Users/andreas/Documents/PostDoc/Clustering/\
#NoiseRecognizer_WorkingVersion/AnalysisDataOrganized/TTX_control_0/"

#OUTPUTFOLDER = BASEFOLDER+"Output/"

#PARAMETERFILE = BASEFOLDER+'Input/PARAMETERS_splitInOutCell.json'

#if not os.path.exists(OUTPUTFOLDER):
#    os.makedirs(OUTPUTFOLDER)

#*********************************************
# Parameters
#*********************************************

# if os.path.isfile(PARAMETERFILE):
#     with open(PARAMETERFILE) as json_file:
#         PARAMETERS = json.load(json_file)
# else:

#     PARAMETERS = {"quantile_of_nonzero":True,\
#                     "intensity_quantile_cutoff":0.9,\
#                     "sigma_gaussian_filter":10,\
#                     "N_x":1000,\
#                     "N_select":20000,\
#                     "pad_cell_border":20}# in pixels (see N_x for pixel number in x-direction)

#     with open(PARAMETERFILE, 'w') as json_file:
#         json.dump(PARAMETERS, json_file, indent=4)

#locals().update(PARAMETERS)
#datascale     :158,

#*********************************************
