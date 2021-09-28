from ClusterBasing import ClusterBasing

basefolder = '/Users/andreas/Documents/NoiseRecognizer_WorkingVersion/';

#    parameterfile = 'ProteinData_ttx_1hr_2/Analysis_dataWindow_3/dataWindow_3_parameters';
#    parameterfile = 'ProteinData_ttx_1hr_2/Analysis_dataWindow_7/dataWindow_7_parameters';
parameterfile = 'MikeData/Analysis_dataWindow_1/dataWindow_1_parameters';

CB = ClusterBasing(basefolder,parameterfile);
CB.GetClusterings();
