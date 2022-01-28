#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use trained models to evaluate data prepared by 1.0_Data_Preparation.py

@author: dave
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.models import load_model
import time
from natsort import natsorted
import gc
import csv
import seaborn as sns
from random import shuffle
import json
from joblib import Parallel, delayed
#from joblib.pool import has_shareable_memory
import multiprocessing

proc_wd = os.path.dirname(os.path.abspath(__file__))
print(proc_wd)
if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print('Changed working directory to ' + proc_wd)

from CAML import FuncEtc as fn_etc
from CAML import FuncNormalizeInput as fn_normalize
from CAML import FuncClusterClub as fn_cluclu


# ========================================================================
# Email Notification
# ========================================================================
# Requires a valid email_secrets.json file: open email_secrets_DIST.json and save
# it as email_secrets.json. Fill out the settings with your own and save again.
DoEmailNotifications = 'no' # 'yes' Emails will be sent when this stage ends.
                             # 'no'  Emails will not be sent.
                             # 'ask' Ask what to do for each run.


# ========================================================================
# Image Saving
# ========================================================================
# Application of any of the following options also depends on the image having had
# SaveMyImages enabled in its JSON file during Data Preparation.
# If you want to override the JSON setting and save either images for either all or
# none of the files then use the OverrideJSON setting at the end of this section.

ExportImageFormat = ['png'] # Format(s) to save images (given as a Python list).
                             # Format: 'eps', 'png', 'jpg', 'svg'
                             # Add multiple formats to the list and each image will
                             # be saved in that format. For example:                             
                             # ExportImageFormat = ['eps','png'] saves EPS and PNG.

DoImage_ScoresHistogram = False   # Distribution of clustering probabilities for each image
Image_ScoresHistogram_Folder = 'Score Histograms'

# Plot each point coloured by model score or classification label
DoImage_ByModelScores = False
DoImage_ByLabels = False
Image_ScoresLabels_Folder = 'Scores and Labels' # Folder name for the above images

# Heatmap plots of the nearest neighbour distances. Plot a random set of 100 points
# and their NN points' distances. If ground-truth is known this will be used too.
DoImage_InputValues = False        
Image_InputValues_Folder = 'Input Samples'
GroundTruthCol = None # Ground truth data column number, or None to skip 
                      # any ground-truth comparison on plots.

## Dark scheme, good for presentations
#plotfgcolor = (1,1,1)                  # text & line colour for all plots
#plotbgcolor = (0.1765,0.1765,0.1765)    # background colour for all plots

# Light scheme, good for printing and publications
plotfgcolor = (0,0,0)                  # text & line colour for all plots
plotbgcolor = (1,1,1)                   # background colour for all plots

cmapfile = 'CAML/cmaps/candybright_256_rgb.txt' # linear colourmap, e.g. for model score map
cmap2chfile = 'CAML/cmaps/two_channel_bluyel.txt' # two-phase colormap, e.g. for binary classification map
cmap3chfile = 'CAML/cmaps/three_channel_blu_ora_mag.txt' # two-phase colormap, e.g. for binary classification map

# Changed your mind on how you want images saved since doing Stage 1?
# You can do that here.
OverrideJSON = False        # True = Override the JSON setting for each file
                            # False = Use the JSON value for SaveMyImages.
                            
# If you override the JSON setting then the following will be applied instead:
OverrideSaveMyImages = False # True = save images for every file in the queue.
                            # False = save no images for any file in the queue.

# Instructions for saving images at each stage are given by each input-image's 
# 'SaveMyImages' parameter, in its JSON file. If the JSON file does not have a 
# SaveMyImages value (older versions did not use it)then it will be added according
# to the value below.
SaveMyImagesForOldVersions = False

# ========================================================================
# Queue Skipping
# ========================================================================
# Begin processing at this file in the list of files. Should normally be zero to
# start with the first file but you can jump ahead if you wish to resume processing
# an earlier set or if you are batch-processing across machines.
# NB: remember Python's zero-based indexing. To process the first file in the list
# this needs to be set to zero.
starting_index = 0   # normally zero

# End processing at this file index in the list of files. Should normally be zero
# to process all files in the list but you can terminate the list early, e.g.
# to only process a subset of the files or if you are batch-processing across 
# multiple machines
# NB: remember Python's zero-based indexing.
finishing_index = 0  # normally zero

# NB starting_index can be greater than finishing_index; the script will just
# process the files in the reverse order. e.g.
#  FileIndices = [0,1,2,3,4,5,6,7,8,9]
#  starting_index = 0  and finishing_index = 5 >>> Processes Files 0,1,2,3,4
#  starting_index = 8  and finishing_index = 3 >>> Processes Files 8,7,6,5,4


# ========================================================================
# Cluster forming
# ========================================================================
pred_threshold = 0.5 # points need to be above this score to qualify as clustered (normally 0.5).
# This is only for binary classification models (e.g. clustered or not-clustered)
# For multiple classification, the label with the highest score is assigned to the point.

# clusters are formed from clustered-points (as found by the model) which are not
# near to non-clustered points (also found by the model). Clusters need to have
# this many points (or more) to be keep as actual clusters. Usually this is three to
# avoid forming clusters with zero area (lines and points won't form polygons)
minimum_points_for_cluster = 3
# NB: this is for the initial formation of clusters; you may want to form clusters
# using the default value of 3 points-per-cluster and then exclude clusters later
# when you do your final analysis.

# variables for the shape-fitting around clustered points
BufferEps = 'nnd'     # nnd   size is the nn distance for each point
                    # dens  size is based on the point density (same for all points)
                    # dims  size based on the area of bounding box enclosing points
                    # 123   fixed integer value for all points (in nm)
DoErodeBuffer = True  # after shape-union using BufferEps, shape will be shrunk to tighten the outline
DropSmallHoles = True # small holes within the cluster will be closed.


# ========================================================================
# Parallel Processing
# ========================================================================
# for parallel processing. Set total_cpus = 1 to disable parallel processing.
total_cpus = multiprocessing.cpu_count() - 2


# ========================================================================
# Begin main
# ========================================================================
print('Checkpoint 1');

if True:#__name__ == '__main__':

    cmap = fn_etc.load_colormap(cmapfile, flip=False)
    cmap2ch = fn_etc.load_colormap(cmap2chfile, flip=False)
    cmap3ch = fn_etc.load_colormap(cmap3chfile, flip=False)

    # get the input data from this folder by checking for a previous output folder from Stage 1 (distance measurement)
    if 's1_prep_outputpath' in locals():
        # output exists from previous script (data-prep PKL files)
        default_input_path = s1_prep_outputpath
    else:
        default_input_path = ''
        
    # inputpath_novel = fn_etc.askforinput(
    #         message = 'Folder with prepared files (.MemMap and .json)',
    #         errormessage= 'The folder you provided does not exist or you have supplied a file instead of a folder.',
    #         defaultval= default_input_path,
    #         isvalid = lambda v : os.path.isdir(v))
    inputpath_novel = default_input_path;

    if 'model_fname' in locals():
        default_model_file = model_fname # model_fname exists from previous script
    else:
        default_model_file = '' #os.path.abspath(os.path.join(inputpath_novel,'..','3_models/Model_9IUMGY_Raw-Differences_200.0k-Train_30.0k-Val_0.5-Split.h5'))

    # model_fname = fn_etc.askforinput(
    #         message = 'Path to the model file',
    #         errormessage= 'The file you provided does not exist, or the file is not a \'.h5\' file, or you have supplied a folder instead of a file.',
    #         defaultval= default_model_file,
    #         isvalid = lambda v : os.path.isfile(v) and os.path.splitext(v)[1]=='.h5')
    model_fname = default_model_file;

    # extract the model ID
    model_name = os.path.splitext(os.path.basename(os.path.normpath(model_fname)))[0]
    
    # extract the preparation type required for this model from the model_name
    DistancesProcessedAs = model_name.split()[2]
    ModelShortName = model_name.split()[0]
   
    # specify an output folder ... default is to step up one level from the input so we keep things together instead of nested
    # outputpath_novel = fn_etc.askforinput(
    #         message = 'Output folder',
    #         errormessage= 'An output folder name must be specified!',
    #         defaultval= os.path.abspath(os.path.join(inputpath_novel, '..', '4_evaluated_by_' + ModelShortName)),
    #         isvalid = lambda v : len(v) > 0 and not v.isspace())
    outputpath_novel = os.path.abspath(os.path.join(inputpath_novel, '..', '4_evaluated_by_' + ModelShortName));

    print('Checkpoint 2');
    ## get a list of the files to process from the given folder
    dist_files_novel = natsorted([i for i in os.listdir(inputpath_novel) if 'Dists.MemMap' in i])
    data_files_novel = natsorted([i for i in os.listdir(inputpath_novel) if 'Data.MemMap' in i])
    json_files_novel = natsorted([i for i in os.listdir(inputpath_novel) if '.json' in i])

    if np.shape(dist_files_novel)[0] > 0:
        if np.shape(dist_files_novel)[0] == np.shape(data_files_novel)[0] == np.shape(json_files_novel)[0]:
            total_files_novel = np.shape(dist_files_novel)[0]
            print('Found ' + str(total_files_novel) + ' datasets to work with...')
        else:
            # raise ValueError('Files are mismatched! Each dataset requires a -Data.MemMap, -Dists.MemMap, and .json file to proceed.')
            # Process the most complete set that we can find
            go_with = np.argmin((np.shape(dist_files_novel)[0],np.shape(data_files_novel)[0], np.shape(json_files_novel)[0]))
            if go_with == 0 or go_with == 1:
                #data_files or dist_files are the shortest list
                raise ValueError('Files are mismatched! Each dataset requires a -Data.MemMap, -Dists.MemMap, and .json file to proceed.')
            elif go_with == 2:
                #json_files is shortest
                # this should always be the case as it is generated after the other two files!
                fn_etc.info_msg('Processing partial list of files from selected folder -- rerun this script when additional file-sets are added.')
                dist_files_novel = dist_files_novel[:np.shape(json_files_novel)[0]]
                data_files_novel = data_files_novel[:np.shape(json_files_novel)[0]]
                total_files_novel = np.shape(dist_files_novel)[0]
                print('Found ' + str(total_files_novel) + ' datasets to work with...')
    else:
        raise ValueError('No files to process! Check you have a) given the correct folder and b) this folder contains files prepared by the \'Data preparation\' processing step.')


    # load the model
    if os.path.exists(model_fname):

        model = load_model(model_fname)
        model_ID = os.path.basename(model_fname).split()[0]
        print('Loaded Model ' + model_ID + '\t(You can safely ignore warnings above about \'No training configuration found\')')
        model_config = model.get_config()
        print(model)
        print(model_config);
        RequiredInputSize = model_config['layers'][0]['config']['batch_input_shape'][1]
        ModelLabelsTotal = model_config['layers'][-1]['config']['units']
        if ModelLabelsTotal == 1:
            ModelLabelsTotal = 2
        print('This model expects input from ' + str(RequiredInputSize) + ' near-neighbours.')
        
        # check out input folder for clues that it will work with our model
        # if we can't tell from the folder name then we try again later after loading the data
        try:
            InputFolder_Check_A = inputpath_novel.split('_')[-2] # the second-last element should have the NN count
            InputFolder_Check_B = InputFolder_Check_A.split('-')[-1] # Get the last digit
        except:
            InputFolder_Check_A = 'not a folder made by CAML'
            InputFolder_Check_B = str(RequiredInputSize) # if we can't extract the information we need
        InputFolder_Ints=''
        InputFolder_Chars=''
        
        # Separate the chars from the integers. We are expecting cellID(n) here for our simulated cells.
        for s in InputFolder_Check_A:
            if not s.isdigit():
                InputFolder_Chars = InputFolder_Chars + s
                
        for s in InputFolder_Check_B:
            if s.isdigit():
                InputFolder_Ints = InputFolder_Ints + s
        
        if not int(InputFolder_Ints) == RequiredInputSize and InputFolder_Chars == 'dNN(-)':
            raise ValueError('Your input data uses ' + InputFolder_Ints + ' near-neighbour values but Model ' + model_ID + ' is expecting ' + str(RequiredInputSize) + ' near-neighbour values.\r\nPlease use a model which matches the input data size.')

    else:
        raise ValueError('You must supply a valid model file.')
    
    
    # check the starting_index value in case we are restarting a run
    if starting_index != 0:
        keep_starting_index = fn_etc.askforinput(
            message = 'Staring Index is set to ' + str(starting_index) + ' (begin with File ' + str(starting_index+1) + '). Is this correct? Y to carry on or N to start with the first file.',
            errormessage= 'Type Y or N',
            defaultval= 'Y',
            isvalid = lambda v : v.lower() in ['y','n','yes','no'])
        
        if keep_starting_index.lower() in ['y','yes']:
            print('Keeping the current start file. Processing will start with File ' + str(starting_index + 1))            
        else:
            print('Starting point has been reset. Processing will begin with the first file in the list.')
            starting_index = 0
  
    
    # check the finishing_index value in case we are restarting a run
    if finishing_index != 0:
        keep_finishing_index = fn_etc.askforinput(
            message = 'Finishing Index is set to ' + str(finishing_index) + ', i.e. end processing after File ' + str(finishing_index - 1) + ' is done. Is this correct? Y to carry on or N to end with the last file.',
            errormessage= 'Type Y or N',
            defaultval= 'y',
            isvalid = lambda v : v.lower() in ['y','n','yes','no'])
    
        if keep_finishing_index.lower() in ['y','yes']:
            print('Keeping the current end file. Processing will end once File ' + str(finishing_index) + ' is done.')            
        else:
            finishing_index = total_files_novel
            print('Finishing point has been reset. Processing will end with the last file in the list.')
    else:
        finishing_index = total_files_novel


    if DoEmailNotifications == 'ask':

        choose_email_notification = fn_etc.askforinput(
            message = 'Send a notification email when this stage has finished? (Y or N)',
            errormessage= 'Enter Y or N',
            defaultval= 'n',
            isvalid = lambda v : v.lower() in ['y','n','yes','no'])
    
        if choose_email_notification.lower() in ['y','yes']:
            EmailUponCompletion = True
            print('Notification emails will be sent (if email configuration is correctly set up).')
        else:
            EmailUponCompletion = False
            print('Email notifications will not be sent.')

    elif DoEmailNotifications == 'yes':
        EmailUponCompletion = True
    else:
        EmailUponCompletion = False


    # proceed_with_processing = fn_etc.askforinput(
    #     message = 'When you are ready to proceed press Enter (or X to cancel everything and exit)',
    #     errormessage= 'Type P to proceed or X to exit',
    #     defaultval= 'P',
    #     isvalid = lambda v : v.lower() in ['p','x'])
    proceed_with_processing = 'P';
    
    if proceed_with_processing.lower() in ['p']:
        print('Rightyo, off we go...\n')
        if total_cpus > 1 and os.name == 'posix':
            multiprocessing.set_start_method('forkserver', force=True)
    elif proceed_with_processing.lower() in ['x']:
        print('That\'s ok. Maybe next time?')
        raise ValueError('Not an error, you just decided not to proceed and that\'s OK! :)')

    #make the folder for the output data
    if not os.path.exists(outputpath_novel):
        os.makedirs(outputpath_novel)

    # make a folder for the Histogram of Model Scores (if required)
    if DoImage_ScoresHistogram and not os.path.exists(os.path.join(outputpath_novel, Image_ScoresHistogram_Folder)):
        os.makedirs(os.path.join(outputpath_novel, Image_ScoresHistogram_Folder))

    # make a folder for the Input Values heatmap (if required)
    if DoImage_InputValues and not os.path.exists(os.path.join(outputpath_novel, Image_InputValues_Folder)):
        os.makedirs(os.path.join(outputpath_novel, Image_InputValues_Folder))



    # open a log file to keep track of model
    ModelLog_fname = 'Model ' + model_ID + ' Eval Log.txt'
    with open(os.path.join(outputpath_novel, ModelLog_fname), 'a+', newline='') as model_log:
        writer = csv.writer(model_log, delimiter='\t')
        writer.writerow(['FileID', 'Total Points', 'Model Evaluation Time (s)', 'Cluster Segmentation Time (s)', 'Cluster Shape Fitting Time (s)'])

    print('Checkpoint 3');
    # ========================================================================
    #    For each file ... 
    # ========================================================================
    #        
    # using starting_index we can easily pick up again in case of stoppages
    for fileIdx in range(starting_index, finishing_index):

        starting_index = fileIdx # update starting index in case something goes wrong and we need to restart this loop here directly
        current_file = json_files_novel[fileIdx]

        # load this file's ProcSettings from json
        with open(os.path.join(inputpath_novel, current_file), 'r') as file:
            ps_current = json.loads(file.read())

        # older processed files did not use this but we can add it in at this stage
        if 'SaveMyImages' not in ps_current:
            ps_current['SaveMyImages'] = SaveMyImagesForOldVersions
        
        # older versions did not use this flag. Add it here if it's missing.
        if 'three_dee' not in ps_current:
            ps_current['three_dee'] = False
        
        # older versions had a different ImageSize format. Update it here.
        if type(ps_current['ImageSize']) == int:
            ps_current['ImageSize'] = [ps_current['ImageSize'], ps_current['ImageSize'], 0]

        if OverrideJSON:
           ps_current['SaveMyImages'] = OverrideSaveMyImages

        # make a folder for the Scores and Labels scatterplots (if required)
        if ps_current['SaveMyImages'] and fileIdx == starting_index:
            if DoImage_ByModelScores or DoImage_ByLabels:
                if not os.path.exists(os.path.join(outputpath_novel, Image_ScoresLabels_Folder)):
                    os.makedirs(os.path.join(outputpath_novel, Image_ScoresLabels_Folder))
            
        datatable_called_fname = os.path.join(outputpath_novel, ps_current['FilePrefix'] + '_DataCalled' + ps_current['InputFileExt'])

        fn_etc.progress_msg(str(fileIdx + 1) + ' of ' + str(finishing_index), ps_current['FilePrefix'])

        if os.path.exists(datatable_called_fname):
            print('SKIPPED: a finished output file already exists.')
        else:
            # load from memmap file
            print('Loading data... ', end='', flush=True)
            import_dists_f = os.path.join(inputpath_novel, ps_current['dists_mmap_fname'])
            if os.path.isfile(import_dists_f):
                Dists_all_New = np.memmap(import_dists_f, dtype='float64', shape=tuple(ps_current['DistsDumpShape']), mode='r')

            if ps_current['ClusMembershipIDCol']:
                import_vectors_f = os.path.join(inputpath_novel, ps_current['target_vectors_mmap_fname'])
                target_vectors = np.memmap(import_vectors_f, dtype='int', shape=tuple(ps_current['VectorsDumpShape']), mode='r')
                
                import_binary_f = os.path.join(inputpath_novel, ps_current['target_binary_mmap_fname'])
                target_binary = np.memmap(import_binary_f, dtype='int', shape=tuple(ps_current['BinaryDumpShape']), mode='r')
            print('Done.')

            TotalPointsThisImage = Dists_all_New.shape[0]
            print('Total of ' + str(TotalPointsThisImage) + ' points to evaluate.')

            # check that we are using the right model for this data
            TotalNeighboursThisImage = Dists_all_New.shape[1]
            if TotalNeighboursThisImage != RequiredInputSize:
                raise ValueError('Your input data uses ' + str(TotalNeighboursThisImage) + ' near-neighbour values but Model ' + model_ID + ' is expecting ' + str(RequiredInputSize) + ' near-neighbour values.\r\nPlease use a model which matches the input data size.')

            # ========================================================================
            #  Model Preparation: Turn raw input (distances) into input for the model
            # ========================================================================

            X_novel_distances_raw = np.array(Dists_all_New[:,:,0], dtype='float32') # convert from float64 to float32 to ease memory requirements            
            X_novel = fn_normalize.normalize_dists(X_novel_distances_raw, TotalPointsThisImage, DistancesProcessedAs)
            if ModelLabelsTotal == 2:
                X_novel = X_novel.reshape((X_novel.shape[0],X_novel.shape[1], 1)) # Reshape the data to be repeats/measures/features for LSTM

            _ = gc.collect()

            ### make predictions
            modeleval_time_init = time.time()
            
            novel_probabilities = model.predict(X_novel, batch_size=64, verbose=1)       # probability assessment of event being in cluster(~1) or not (~0)
            if ModelLabelsTotal == 2:
                novel_predictions = [float(np.round(x - (pred_threshold - 0.5))) for x in novel_probabilities]   # convert probability into boolean by pessimistic threshold ... must be >pred_threshold to qualify as 1
            else:
                novel_predictions = novel_probabilities.argmax(axis=-1) # convert model's probabilities to labels

            # at this stage we should delete and salted points which have been labelled as clustered
            # idx_salt = 
            # idx_clustered = 
            # idx_salty_clustered = 
            # novel_probabilities = not salty_clustered
            # novel_predictions = not salty_clustered
            
            modeleval_time = time.time() - modeleval_time_init
            print('Time with model: ' + str(round(modeleval_time,2)) + ' seconds for ' + str(TotalPointsThisImage) + ' points.')
            
            pickle.dump((novel_probabilities, novel_predictions),open(os.path.join(outputpath_novel, ps_current['FilePrefix'] + '_ML_calls[NN' + str(ps_current['ClosestFriend']) + '-' + str(ps_current['FurthestFriend']) + '].pkl'), 'wb'))

            _ = gc.collect()
        
            # load from memmap file
            import_datatable_f = os.path.join(inputpath_novel, ps_current['datatable_mmap_fname'])
            if os.path.isfile(import_datatable_f):
                datatable_mmapped = np.memmap(import_datatable_f, dtype='float64', shape=tuple(ps_current['DataDumpShape']), mode='r')

            datatable_called = np.concatenate((datatable_mmapped, novel_probabilities,np.array(novel_predictions).reshape(TotalPointsThisImage,1)),axis=1)
            ps_current['TableHeaders'] = ps_current['TableHeaders'] + ps_current['InputFileDelimiter'] + 'score' + ps_current['InputFileDelimiter'] + 'label (' + model_ID + ')'

            scores_col = datatable_called.shape[1] - 2
            labels_col = datatable_called.shape[1] - 1
            
            # some stats about this thing
            TotalType0Points = sum([idx == 0 for idx in novel_predictions])
            TotalType1Points = sum([idx == 1 for idx in novel_predictions])
            if ModelLabelsTotal == 3:
                TotalType2Points = sum([idx == 2 for idx in novel_predictions])
                PercentClustered = ((TotalType1Points + TotalType2Points) / TotalPointsThisImage) * 100
            else:
                PercentClustered = ((TotalType1Points) / TotalPointsThisImage) * 100
            
            print('Model ' + model_ID + ' thinks File ' + str(fileIdx + 1) + ' has ' + str(round(PercentClustered, 2)) + ' percent of points in clusters')
            if ModelLabelsTotal == 3:
                print('\t' + str(round(TotalType1Points/TotalPointsThisImage*100, 2)) + ' percent of points are Type 1')
                print('\t' + str(round(TotalType2Points/TotalPointsThisImage*100, 2)) + ' percent of points are Type 2')

            # Save histogram of probability distribution
            if DoImage_ScoresHistogram:
                plt.ioff()
                sns_fig, sns_ax = plt.subplots(figsize=(10, 10))
                if ModelLabelsTotal == 2:
                    sns_plot = sns.distplot(novel_probabilities, kde=False, ax=sns_ax)
                    sns_plot.set(xlabel='\'In Cluster\' Probability', ylabel='Frequency (Points)', xlim=[0, 1])
                else:
                    sns_plot = sns.distplot(novel_probabilities, kde=False, ax=sns_ax, color=['orange','cyan','magenta'], label=['Type 0', 'Type 1', 'Type 2'])
                    sns_plot.set(xlabel='Point Type Probability', ylabel='Frequency (Points)', xlim=[0, 1])
                    sns_ax.legend(['Type 0',' Type 1','Type 2'])
                sns_plot.set_title(ps_current['FilePrefix'], fontsize=7)
#                sns_fig = sns_plot.get_figure()
                for ExportImage in ExportImageFormat:
                    sns_plot_fname = os.path.join(outputpath_novel, Image_ScoresHistogram_Folder, ps_current['FilePrefix'] + ' - Histogram.' + ExportImage)
                    sns_fig.savefig(sns_plot_fname)
#                sns_fig.savefig(sns_plot_fname + '.png')
#                sns_fig.savefig(sns_plot_fname + '.svg')
                plt.close('all')
                plt.ion()
            
            print('step a');
            # Save scatter images coloured by probability or clustering
            # auto image boundaries
            if ps_current['AutoAxes'] == True:
                ps_current['xMin'] = np.floor(np.min(datatable_mmapped[:,ps_current['xCol']]) / ps_current['AutoAxesNearest'] ) * ps_current['AutoAxesNearest']
                ps_current['xMax'] = np.ceil(np.max(datatable_mmapped[:,ps_current['xCol']]) / ps_current['AutoAxesNearest'] ) * ps_current['AutoAxesNearest']
                ps_current['yMin'] = np.floor(np.min(datatable_mmapped[:,ps_current['yCol']]) / ps_current['AutoAxesNearest'] ) * ps_current['AutoAxesNearest']
                ps_current['yMax'] = np.ceil(np.max(datatable_mmapped[:,ps_current['yCol']]) / ps_current['AutoAxesNearest'] ) * ps_current['AutoAxesNearest']               

            if ps_current['SaveMyImages']:
            
                if DoImage_ByModelScores or DoImage_ByLabels:
                    ps_temp = ps_current
                    ps_temp['BackGroundColor'] = plotbgcolor
                    ps_temp['ForeGroundColor'] = plotfgcolor
                    if ps_temp['ImageSize'][0] /50 < 300:
                        ps_temp['OutputScaling'] = 1
                        ps_temp['PointSize'] = 10
                    else:
                        ps_temp['OutputScaling'] = 100
                        ps_temp['PointSize'] = round(ps_temp['ImageSize'][0] / 5000) + 1
    
                if DoImage_ByModelScores:
                    ps_temp['ClusMembershipIDCol'] = scores_col
                    ps_temp['PointsMinValue'] = 0
                    ps_temp['PointsMaxValue'] = 1
                    print('Saving image of points coloured by model\'s score... ', end='', flush=True)
                    for ExportImage in ExportImageFormat:
                        fn_etc.make_image(datatable_called, os.path.join(outputpath_novel, Image_ScoresLabels_Folder, ps_current['FilePrefix'] + ' - Model Scores.' + ExportImage), ps_temp, cmap)
                    print('Done.')
                    
                if DoImage_ByLabels:
                    ps_temp['ClusMembershipIDCol'] = labels_col
                    ps_temp['PointsMinValue'] = 0
                    ps_temp['PointsMaxValue'] = ModelLabelsTotal - 1
                    print('Saving image of points colours by classification label... ',end='', flush=True)
                    for ExportImage in ExportImageFormat:
                        if ModelLabelsTotal == 2:
                            fn_etc.make_image(datatable_called, os.path.join(outputpath_novel, Image_ScoresLabels_Folder, ps_current['FilePrefix'] + ' - Classification Labels.' + ExportImage), ps_temp, cmap2ch)
                        else:
                            fn_etc.make_image(datatable_called, os.path.join(outputpath_novel, Image_ScoresLabels_Folder, ps_current['FilePrefix'] + ' - Classification Labels.' + ExportImage), ps_temp, cmap3ch)
                    print('Done.')

                # plot distance-input heatmaps for each class of point identified by the model
                # recycles the input data as fed to the model
                if DoImage_InputValues:

                    # Set these both to pred_threshold to sample from the full range of available points for each label
                    # otherwise set these to specific values to narrow down the range of points used in the plot
                    inputplot_clus_minscore = pred_threshold
                    inputplot_clus_maxscore = 1.0
                                    
                    inputplot_nonclus_minscore = 0.0
                    inputplot_nonclus_maxscore = pred_threshold
                    
                    clustered_indices = np.where(np.logical_and(datatable_called[:, labels_col] > 0, datatable_called[:, scores_col] > inputplot_clus_minscore, datatable_called[:, scores_col] <= inputplot_clus_maxscore))[0]
                    nonclustered_indices = np.where(np.logical_and(datatable_called[:, labels_col] == 0, datatable_called[:, scores_col] >= inputplot_nonclus_minscore, datatable_called[:, scores_col] < inputplot_nonclus_maxscore))[0]

                    if GroundTruthCol != None:
                        ground_truth = datatable_called[:, GroundTruthCol] > 0

                    # shuffle the collection of indices for each category
                    shuffle(clustered_indices)
                    shuffle(nonclustered_indices)
                    
                    # keep the first 100
                    clustered_indices = clustered_indices[0:100]
                    nonclustered_indices = nonclustered_indices[0:100]
                    
                    # get overall min/max values
                    colormin = np.floor(np.min((X_novel[clustered_indices,:,0].min(), X_novel[nonclustered_indices,:,0].min())))
                    colormax = np.ceil(np.max((X_novel[clustered_indices,:,0].max(), X_novel[nonclustered_indices,:,0].max())))

                    colour_axis_label = DistancesProcessedAs + ' input values'
                    
                    plt.ioff()
                    
                    if GroundTruthCol != None:
                        fig_dd, (ax_nclu, ax_nclu_gt, ax_clus, ax_clus_gt) = plt.subplots(1, 4, gridspec_kw = {'width_ratios':[30, 1, 30, 1]})
                    else:
                        fig_dd, (ax_nclu, ax_clus) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[1, 1]})

                    # plot the nonclustered points
                    nclu_distdiffsmat = ax_nclu.matshow(np.array(X_novel[nonclustered_indices,:,0], dtype='float32'), cmap=cmap, vmin=colormin, vmax=colormax)
                    
                    # fix the line colours to match specified colours
                    ax_nclu.tick_params(color=plotfgcolor, labelcolor=plotfgcolor)
                    for spine in ax_nclu.spines.values():
                        spine.set_edgecolor(plotfgcolor)    

                    # add axis labels
                    ax_nclu.set_title('100 Random Non-clustered Points (scores < ' + str(inputplot_nonclus_maxscore) + ')', color=plotfgcolor, y=1.1)
                    ax_nclu.set_xlabel('nth Nearest Neighbour', color=plotfgcolor)
                    ax_nclu.xaxis.set_label_position('top') 
                    ax_nclu.set_ylabel('Point #', color=plotfgcolor)

                    # add nclu colorbar
                    ax_nclu_cbar = fig_dd.colorbar(nclu_distdiffsmat, fraction=0.046, pad=0.04, ax = ax_nclu)
                    ax_nclu_cbar.set_label(colour_axis_label, color=plotfgcolor)  # set colorbar label plus label color
                    ax_nclu_cbar.ax.yaxis.set_tick_params(color=plotfgcolor)     # set colorbar tick color
                    ax_nclu_cbar.outline.set_edgecolor(plotfgcolor)              # set colorbar edgecolor 
                    plt.setp(plt.getp(ax_nclu_cbar.ax.axes, 'yticklabels'), color=plotfgcolor)

                    if GroundTruthCol != None:
                        # plot the nonclustered points' ground truth
                        nclu_grndtruthsmat = ax_nclu_gt.matshow(np.array(ground_truth[nonclustered_indices, None], dtype='float32'), cmap=cmap2ch, vmin=0, vmax=1)
                        ax_gtnclu_cbar = fig_dd.colorbar(nclu_grndtruthsmat, fraction=0.46, pad=0.15, ax=ax_nclu_gt)

                    # plot the clustered points
                    clus_distdiffsmat = ax_clus.matshow(np.array(X_novel[clustered_indices,:,0], dtype='float32'), cmap=cmap, vmin=colormin, vmax=colormax)

                    # fix the line colours to match specified colours
                    ax_clus.tick_params(color=plotfgcolor, labelcolor=plotfgcolor)
                    for spine in ax_clus.spines.values():
                        spine.set_edgecolor(plotfgcolor)

                    # add axis labels
                    ax_clus.set_title('100 Random Clustered Points (scores > ' + str(inputplot_clus_minscore) + ')', color=plotfgcolor, y=1.1)
                    ax_clus.set_xlabel('nth Nearest Neighbour', color=plotfgcolor)
                    ax_clus.xaxis.set_label_position('top') 
                    ax_clus.set_ylabel('Point #', color=plotfgcolor)

                    # Add colourbar
                    ax_clus_cbar = fig_dd.colorbar(clus_distdiffsmat, fraction=0.046, pad=0.04, ax=ax_clus)
                    ax_clus_cbar.set_label(colour_axis_label, color=plotfgcolor)  # set colorbar label plus label color
                    ax_clus_cbar.ax.yaxis.set_tick_params(color=plotfgcolor)     # set colorbar tick color
                    ax_clus_cbar.outline.set_edgecolor(plotfgcolor)              # set colorbar edgecolor 
                    plt.setp(plt.getp(ax_clus_cbar.ax.axes, 'yticklabels'), color=plotfgcolor)

                    if GroundTruthCol != None:
                        # plot the nonclustered points' ground truth
                        clus_grndtruthsmat = ax_clus_gt.matshow(np.array(ground_truth[clustered_indices, None], dtype='float32'), cmap=cmap2ch, vmin=colormin, vmax=colormax)
                        ax_gtclus_cbar = fig_dd.colorbar(clus_grndtruthsmat, fraction=0.46, pad=0.15, ax=ax_clus_gt)

                    for ExportImage in ExportImageFormat:
                        DistancePlots_fname = os.path.join(outputpath_novel, Image_InputValues_Folder, ps_current['FilePrefix'] + 'Differences in Distances - Some Random Points.' + ExportImage)
                        fig_dd.savefig(DistancePlots_fname,
                                    dpi=300,
                                    bbox_inches=0,
                                    facecolor=plotbgcolor,
                                    edgecolor='none',
                                    transparent=True)

                    plt.close()
                    plt.ion()
                    del [clustered_indices, nonclustered_indices]
                ### End of doDistancePlots

            _ = gc.collect()

            print('step b');
            # ========================================================================
            #  Cluster Club: group points into clusters from nearby clustered points
            # ========================================================================

            #use Dists_all_New
            # nth point's data = Dists_all_New[n,:,:]
            # just the distances = Dists_all_New[:,:,0]
            # just the UIDs = Dists_all_New[:,:,-1]
            # just the NN coords = Dists_all_New[:,:,(ps_current['xCol']+1,ps_current['yCol']+1)] e.g. plt.scatter(Just_nnxy[0,:,0],Just_nnxy[0,:,1]) to plot all the NNs.

            # Cluster scores are in novel_probabilities
            # Cluster labels are in novel_predictions

            # a cluster is any group of points (labelled as clustered) whose nearest neighbours are other labelled-as-clustered points (and no non-clustered points are closer)

            # find indicies for the points which are clustered (== 1). Todo: this can be set to specific labels to do clustering for specific types.
            WorkList_onlyclustered = datatable_mmapped[np.array(novel_predictions) == 1, ps_current['UIDCol']].astype(int) - 1
            # add the clustering scores alongside the indices
            WorkList_probs_onlyclustered = novel_probabilities[WorkList_onlyclustered, :]
            WorkList = np.concatenate([WorkList_onlyclustered.reshape([WorkList_onlyclustered.size,1]), WorkList_probs_onlyclustered], axis=1)
            # sort rows by score column; highest score is first
            WorkList = WorkList[WorkList[:, 1].argsort()[::-1]]
            WorkList = WorkList[:,0].astype(int)
            del(WorkList_onlyclustered, WorkList_probs_onlyclustered)

            # new array for each point's cluster membership
            # all points initially receive a ClusterUID of -1 (i.e. not-clustered)
            novel_clusterclub = np.full((TotalPointsThisImage), -1).astype(int)
            
            # the clustered points now receive a starting CluUID which matches 
            # their ranked cluster score (most clustered point = 1)
            novel_clusterclub[WorkList] = np.array(range(1, len(WorkList) + 1)) 

            if ModelLabelsTotal == 2 and TotalType1Points > minimum_points_for_cluster:
                # starting with the most clustered point, accumulate its nearest
                # consecutive also-clustered neighbours into the same clusterUID
                if PercentClustered > 95:
                    fn_etc.progress_msg('***', 'High percentage of clustered points: Performing additional cluster segmentation...')
    
                print('Segmenting ' + str(TotalType1Points) + ' clustered points into like-clusters: ', end='', flush=True)
                
                segmentation_time_init = time.time()
    
                # set up the progress bar
                BarLength = 20          # length of the progress bar
                CompletedBlock = '█'    # completed segments look like this
                RemainingBlock = '░'    # remaining segments look like this
                Flipper_A = '▓'         # the flipper lets you know things are still
                Flipper_B = '▒'         # 'alive' inside!
                # completed blocks will appear when these milestones are reached:
                UpdateProgress_UIDs = list(WorkList[::int(WorkList.shape[0]/BarLength+1)])
                UpdateProgress_UIDs.pop(0) # remove the first index
                UpdateProgress_UIDs.append(WorkList[-1]) # add the last index to the list
                HeartBeat_UIDs = list(set(list(WorkList[::25])).difference(UpdateProgress_UIDs))
                
                # initialise the progress bar and print it
                RemainingLength = BarLength
                RemainingStr = RemainingBlock * RemainingLength
                CompletedLength = 0
                CompletedStr = CompletedBlock * CompletedLength
                DeleteStr = '\b' * (BarLength)
                Flipper = Flipper_A
                print(CompletedStr + RemainingStr, end='', flush=True)
                
                refined_cluster_call = np.array(novel_predictions)
                
                for CurrentPoint_UID in WorkList:
                    
                    CurrentPoint_Label = novel_predictions[CurrentPoint_UID] # ML Label (C or NC)
    
                    # Get list of CurrentPoint's NN-UIDs, subtract one to get their indices
                    CurrPts_NNUIDs = Dists_all_New[CurrentPoint_UID, :, -1].astype(int) - 1
    
                    # check each NN in turn and stop when they no longer match the ML-label of CurrentPoint
                    # i.e. we are finding all our absolute nearest neighbours of the same type.
                    nn_ends_IDx = 0
                    for x in CurrPts_NNUIDs:
                        if refined_cluster_call[x] == CurrentPoint_Label:
                            nn_ends_IDx += 1
                        else:
                            break
                    
                    # only make a cluster if there enough other points to look at
                    if nn_ends_IDx >= minimum_points_for_cluster:
    
                        if PercentClustered > 95:
                            # clusters can be joined if there are no NC points between them
                            # check the distance-differences and limit this cluster to points
                            # with similar distances
                            CurrentDists = np.diff(Dists_all_New[CurrentPoint_UID,:nn_ends_IDx,0]) # / Dists_all_New[CurrentPoint_UID,0,0]
                            CurrDistsMinusMedian = np.abs(CurrentDists - np.median(CurrentDists))
                            CDMedianDev = np.median(CurrDistsMinusMedian)
                            CurrDistsScore = CurrDistsMinusMedian / CDMedianDev if CDMedianDev else 0.0
                            OutlierPoints = np.array(np.where(CurrDistsScore > 5 * CDMedianDev))
                            OutlierDistance = np.shape(OutlierPoints)[1]
    
                            if OutlierDistance > minimum_points_for_cluster:
                                nn_original_IDx = nn_ends_IDx
                                nn_ends_IDx = OutlierPoints[0,0] + 1 # we stop this cluster growing at the first (closest NN) outlier (plus 1 to include the furthest point before the jump)
    
                        # Get the current cluster IDs; excluding -1 IDs
                        Existing_ClusIDs = novel_clusterclub[CurrPts_NNUIDs[0:nn_ends_IDx]]
                        Potential_ClusIDs = Existing_ClusIDs[Existing_ClusIDs > 0,]
    
                        if Potential_ClusIDs.size > 0:
                            # our nearest also-clustered neighbours have usable cluster IDs
                            ImpendingAssignmentID = np.min((np.min(Potential_ClusIDs), novel_clusterclub[CurrentPoint_UID]))
                            novel_clusterclub[CurrentPoint_UID] = ImpendingAssignmentID
                            novel_clusterclub[CurrPts_NNUIDs[0:nn_ends_IDx]] = ImpendingAssignmentID
                        else:
                            # our nearest also-clustered neighbours have been demoted during an earlier iteration
                            # we have no useable IDs so this point is also demoted. Sorry about that, point.
                            novel_clusterclub[CurrentPoint_UID] = -1 # assign NC membership
                            refined_cluster_call[CurrentPoint_UID] = 0  # update label to NC
                    else:
                        novel_clusterclub[CurrentPoint_UID] = -1 # demote this point as it doesn't have enough friends to bother with
                        refined_cluster_call[CurrentPoint_UID] = 0  # also update the label to reflect demotion
                
                    # update the progress bar
                    if CurrentPoint_UID in UpdateProgress_UIDs:
                        RemainingLength -= 1
                        RemainingStr = RemainingBlock * RemainingLength
                        CompletedLength += 1
                        CompletedStr = CompletedBlock * CompletedLength
                        print(DeleteStr + CompletedStr + RemainingStr, end='', flush=True)
                    
                    if CurrentPoint_UID in HeartBeat_UIDs:
                        if Flipper == Flipper_A:
                            Flipper = Flipper_B
                        elif Flipper == Flipper_B:
                            Flipper = Flipper_A
                        RemainingStr = RemainingBlock * (RemainingLength - 1)
                        print(DeleteStr + CompletedStr + Flipper + RemainingStr, end='', flush=True)
    
                segmentation_time = time.time() - segmentation_time_init
                # End of processing each point in WorkList
    
                # Initial summary
                UniqueClusIDs = np.unique(novel_clusterclub)
                UniqueClusIDs = UniqueClusIDs[UniqueClusIDs != -1] # exlude ID = -1 (nonclustered)
    
                total_clusters = np.shape(UniqueClusIDs)[0]
                
                # done processing - finish the progress bar
                print(' Done (found ' + str(total_clusters) + ' clusters)', flush=True)
    
    #            # we can now remove the BenDay points as they have done their job
    #            original_points = np.sum(~np.isnan(datatable_mmapped[:,0])) # assumes column 0 has some trivial data to check for NaNs!!
    #            Dists_all_New = Dists_all_New[:original_points, :, :]
    #            novel_clusterclub = novel_clusterclub[:original_points, ]
    #            datatable_mmapped = datatable_mmapped[:original_points, :]
    #            ps_current['DistsDumpShape'][0] = original_points
    #            ps_current['DataDumpShape'][0] = original_points
                
                # ========================================================================
                #  Cluster Shapes: Create a polygon containing like-clustered points
                # ========================================================================
                
                if not ps_current['three_dee']:
                    demoted_points = list()     # clustered points to be remarked as not-clustered
                    demoted_clusters = list()   # clusterIDs to delete
                    reassigned_points = list()  # clustered points which have been given new clusterIDs
                    cluclub_out = list()        # cluster info: ClusterID, OutlineCoords, Area, Population (of points)
                    
                    if BufferEps == 'nnd':
                        NNDs = Dists_all_New[:,0,0]
                    else:
                        NNDs = list() # no sense passing a pile of unused data to each worker
                    
                    print('Forming shapes around like-clustered points...')
                    geometryfit_time_init = time.time()
                    
#                    _ = Parallel(n_jobs=total_cpus, verbose=3)(delayed(has_shareable_memory)(fn_cluclu.cluster_shaper(
                    _ = Parallel(n_jobs=total_cpus, verbose=3, require='sharedmem')(delayed(fn_cluclu.cluster_shaper)(
                            i, 
                            novel_clusterclub, 
                            UniqueClusIDs, 
                            datatable_mmapped, 
                            NNDs,
                            BufferEps, 
                            DoErodeBuffer, 
                            DropSmallHoles,
                            demoted_points,
                            demoted_clusters,
                            reassigned_points,
                            cluclub_out,
                            ps_current
                            ) for i in range(0, total_clusters))
        
        #            # Non-parallel equivalent
        #            for i in range(0, total_clusters):
        #                fn_cluclu.cluster_shaper(
        #                    i, 
        #                    novel_clusterclub, 
        #                    UniqueClusIDs, 
        #                    datatable_mmapped, 
        #                    NNDs,
        #                    BufferEps, 
        #                    DoErodeBuffer, 
        #                    DropSmallHoles,
        #                    demoted_points,
        #                    demoted_clusters,
        #                    reassigned_points,
        #                    cluclub_out,
        #                    ps_current
        #                    ) 
        
                    # ========================================================================
                    #  Cleaning: Remove demoted points and sanitise the cluster ID list
                    # ========================================================================
                    
                    # Sort the cluster-info list by ClusterID and unpack into separate lists
                    cluclub_out = sorted(cluclub_out)
                    
                    cluclub_cIDs = list()
                    cluclub_path = list()
                    cluclub_area = list()
                    cluclub_popn = list()
                    cluclub_cntr = list()
                    
                    for clublah in cluclub_out:
                        cluclub_cIDs.append(clublah[0])
                        cluclub_path.append(clublah[1])
                        cluclub_area.append(clublah[2])
                        cluclub_popn.append(clublah[3])
                        cluclub_cntr.append(clublah[4])
                    
                    del cluclub_out
                    
                    # find points with demoted IDs and reassign them as non-clustered
                    for i in demoted_clusters:
                        refined_cluster_call[np.where(novel_clusterclub == i)] = 0 # first flip the status of points with demoted clusters
                        novel_clusterclub[np.where(novel_clusterclub == i)] = -1   # then reassign the NC cluster-ID
                        
        
                    # update the IDs for points in clusters that got split up
                    for reassign_task in reassigned_points:
                        novel_clusterclub[reassign_task[:, 0] - 1] = reassign_task[0, 1] # subtract 1 to convert pointUID to index (zero base indexing)
        
                    # update cluster status of demoted points from C to NC
                    for demoted_pt in demoted_points:
                        refined_cluster_call[demoted_pt - 1] = 0 # update classification (1 --> 0 for demoted points)
                        novel_clusterclub[demoted_pt - 1] = -1  # update clusterIDs (OriginalID --> -1 for demoted points)
        
                    # scan for duplicates in the list of cluster UIDs-- there shouldn't be any!
                    dups = [x for x in cluclub_cIDs if cluclub_cIDs.count(x) > 1]
                    assert len(dups) == 0
        
                    # update the list of unique ClusterIDs after CluClub refinement
                    UniqueClusIDs = np.unique(novel_clusterclub)        
                    UniqueClusIDs = UniqueClusIDs[UniqueClusIDs != -1] # exlude ID = -1 (nonclustered)
                    assert UniqueClusIDs.shape[0] == len(cluclub_cIDs)
                    
                    # if we have changed any points' labels, set a flag here
                    if all(novel_predictions == refined_cluster_call):
                        refined_points_exist = False
                    else:
                        refined_points_exist = True
                    
                else:
                    fn_etc.info_msg('Cluster shape forming is not yet supported for 3D images.')

                ReassignedClusIDs = np.array(range(1, np.shape(UniqueClusIDs)[0] + 1), dtype='int')
                
                # jumble the reassingment order. This helps to 'distance' similarly-numbered clusters 
                # produced by cluster-splitting and such clusters will have contrasting colours on the images.
                # Otherwise two nearby clusters get essentially the same colour
                ReassignedClusIDs = np.random.permutation(ReassignedClusIDs) 
                
#                ReassignedClusIDs[1:-1] = np.random.permutation(ReassignedClusIDs[1:-1]) # Reassign clusters randomly but keep the first entry (the NC UID) intact
    
                novel_clusterclub_harmonised = np.zeros(novel_clusterclub.shape).astype(int)
                for idx, NewClusterID in enumerate(ReassignedClusIDs):
                    novel_clusterclub_harmonised[np.where(novel_clusterclub == UniqueClusIDs[idx])] = NewClusterID
                novel_clusterclub = novel_clusterclub_harmonised
                cluclub_cIDs = ReassignedClusIDs.tolist()
                
                del(novel_clusterclub_harmonised, ReassignedClusIDs)
    
                novel_clusterclub[novel_clusterclub == -1] = 0 # relabel NC points as 0 (rather than -1) for ease of plotting
                
                UniqueClusIDs = np.unique(novel_clusterclub)
                UniqueClusIDs = UniqueClusIDs[UniqueClusIDs != 0] # exlude ID = 0 (nonclustered)
    
                if not ps_current['three_dee']:
                    # Re-assess the total clusters
                    total_clusters = np.shape(UniqueClusIDs)[0]
                    TotalClusteredPoints = sum([idx > 0 for idx in refined_cluster_call])
                    PercentClustered = (TotalClusteredPoints / TotalPointsThisImage) * 100
                    print('File ' + str(fileIdx + 1) + ' has ' + str(round(PercentClustered, 2)) + ' percent of all points in ' + str(total_clusters) + ' clusters (after refinement)')
            
            elif ModelLabelsTotal > 2:
                
                # not enough clustered points to make clusters
                fn_etc.info_msg('Cluster formation for model labels > 2 is not yet supported (you have ' + str(ModelLabelsTotal) + ' labels).')
                total_clusters = -1
                TotalClusteredPoints = -1
                refined_points_exist = False
                
            else:
                
                # not enough clustered points to make clusters
                fn_etc.info_msg('This image has ' + str(TotalType1Points) + ' clustered points and a minimum of ' + str(minimum_points_for_cluster) + ' clustered points are required to form a cluster. Clustering images will not be saved for this image.')
                total_clusters = 0
                TotalClusteredPoints = 0
                refined_points_exist = False            
                
#            if not ps_current['three_dee'] and TotalType1Points > minimum_points_for_cluster:
#                # only log this info for 2D 2-label (for now)                
            geometryfit_time = time.time() - geometryfit_time_init
            with open(os.path.join(outputpath_novel, ModelLog_fname), 'a+', newline='') as model_log:
                writer = csv.writer(model_log, delimiter='\t')
                writer.writerow([str(fileIdx + 1), str(TotalPointsThisImage), str(round(modeleval_time,2)), str(round(segmentation_time,2)), str(round(geometryfit_time,2))])

            # ========================================================================
            #  Export: annotated data table and cluster shape data
            # ========================================================================

            print('Saving new data table with cluster information...', end='', flush=True)
            if refined_points_exist:
                datatable_clusIDs = np.concatenate((datatable_called,
                                                    refined_cluster_call.reshape(TotalPointsThisImage, 1),
                                                    novel_clusterclub.reshape(TotalPointsThisImage, 1)),
                                                    axis=1)
                refined_labels_col = datatable_clusIDs.shape[1] - 2
                clusID_col = np.shape(datatable_clusIDs)[1] - 1
                ps_current['TableHeaders'] = ps_current['TableHeaders'] + ps_current['InputFileDelimiter'] + 'label (refined)' + ps_current['InputFileDelimiter'] + 'clusterID'
            else:
                datatable_clusIDs = np.concatenate((datatable_called,
                                                    novel_clusterclub.reshape(TotalPointsThisImage, 1)),
                                                    axis=1)
                clusID_col = np.shape(datatable_clusIDs)[1]
                ps_current['TableHeaders'] = ps_current['TableHeaders'] + ps_current['InputFileDelimiter'] + 'clusterID'
            
            np.savetxt(datatable_called_fname, datatable_clusIDs, delimiter=ps_current['InputFileDelimiter'], fmt='%10.5f', header=ps_current['TableHeaders'], comments='')
            print('Done.')
            
            if ModelLabelsTotal == 2 and TotalType1Points > minimum_points_for_cluster:
                
                print('Saving cluster shape data...', end='', flush=True)
                # save per-cluster data for this image            
                data_essentials = np.concatenate((datatable_clusIDs[:, ps_current['xCol'], None], datatable_clusIDs[:, ps_current['yCol'], None], datatable_clusIDs[:, clusID_col, None]), axis=1)
                
                cluster_patchlist_fname = os.path.join(outputpath_novel, ps_current['FilePrefix'] + '_ClusterShapes.pkl')
                pickle.dump((cluclub_path, cluclub_popn, cluclub_area, cluclub_cIDs, data_essentials, ps_current), open(cluster_patchlist_fname, 'wb' ), protocol=4)
                print('Done.')
                
                # ========================================================================
                #  Export: high-res images of the full clustering outcome
                # ========================================================================
    
                if ps_current['SaveMyImages'] and not ps_current['three_dee']:
        
                    # make a randomised colour LUT here, same length as the number of unique cluster IDs and with ID=0 as 'grey' for NC points.
                    vals = np.linspace(0, 1, UniqueClusIDs.shape[0]) # include NC points this time
                    np.random.shuffle(vals)
                    jumble_cmap = plt.cm.colors.ListedColormap(cmap(vals))
                    jumble_cmap.set_under(color='grey') # colours outside the scale will be this color.
        
                    print('Saving image of points coloured by cluster ID...', end='', flush=True)
                    ps_temp = ps_current
                    ps_temp['BackGroundColor'] = plotbgcolor
                    if ps_temp['ImageSize'][0] / 50 < 300:
                        ps_temp['OutputScaling'] = 1
                        ps_temp['PointSize'] = 10
                    else:
                        ps_temp['OutputScaling'] = 100
                        ps_temp['PointSize'] = round(ps_temp['ImageSize'][0] / 5000) + 1
                    ps_temp['PointsMinValue'] = np.min(UniqueClusIDs) # minimum needs to be wrt to the cluster IDs so that NC points become out of range and are coloured as 'under'.
                    ps_temp['PointsMaxValue'] = np.max(UniqueClusIDs)
                    ps_temp['ClusMembershipIDCol'] = clusID_col
                    
                    # All points with colour by ClusterID
                    for ExportImage in ExportImageFormat:
                        fn_etc.make_image(datatable_clusIDs, os.path.join(outputpath_novel, Image_ScoresLabels_Folder, ps_current['FilePrefix'] + ' - Cluster Membership.' + ExportImage), ps_temp, jumble_cmap)
                    print('Done.')
        
                    print('Saving image of points with cluster outlines...', end='', flush=True)
        
                    plt.ioff()
                    fig = plt.figure(figsize=(30, 30))
                    ax = fig.add_subplot(111)
                    
                    for patch_idx, patch_p in enumerate(cluclub_path):
                        patch = patches.PathPatch(patch_p, facecolor=jumble_cmap(patch_idx), edgecolor=None, alpha=0.5, lw=0.25, zorder=-1)
                        ax.add_patch(patch)
        
                    plt.scatter(datatable_mmapped[novel_clusterclub > 0, ps_current['xCol']], datatable_mmapped[novel_clusterclub > 0, ps_current['yCol']], s=0.5, c='k', zorder=2, marker='.', edgecolors='none')
                    plt.scatter(datatable_mmapped[novel_clusterclub == 0, ps_current['xCol']], datatable_mmapped[novel_clusterclub == 0, ps_current['yCol']], s=0.5, c='grey', zorder=1, marker='.', edgecolors='none')
                    ax.set_xlim(ps_current['xMin'],ps_current['yMax'])
                    ax.set_ylim(ps_current['yMin'],ps_current['yMax'])
                    plt.rcParams['figure.figsize'] = [30, 30]
                    ax.set_aspect('equal')
                    
                    test_dpi = ps_current['ImageSize'][0] / 50
                    if test_dpi < 300:
                        use_dpi = 300
                    else:
                        use_dpi = test_dpi
    
                    for ExportImage in ExportImageFormat:
                        cluster_outlines_fname = os.path.join(outputpath_novel, Image_ScoresLabels_Folder, ps_current['FilePrefix'] + ' - Cluster Outlines.' + ExportImage)
                        fig.savefig(cluster_outlines_fname,
                                    dpi=use_dpi,
                                    bbox_inches=0,
                                    facecolor=fig.get_facecolor(),
                                    edgecolor='none',
                                    transparent=True)
                    plt.close('all')        
                    plt.ion() # turn on interactive mode
                    print('Done.')


                    # save refined label image
                    if DoImage_ByLabels and refined_points_exist:
                        ps_temp = ps_current
                        ps_temp['BackGroundColor'] = plotbgcolor
                        ps_temp['ForeGroundColor'] = plotfgcolor
                        if ps_temp['ImageSize'][0] /50 < 300:
                            ps_temp['OutputScaling'] = 1
                            ps_temp['PointSize'] = 10
                        else:
                            ps_temp['OutputScaling'] = 100
                            ps_temp['PointSize'] = round(ps_temp['ImageSize'][0] / 5000) + 1
                        ps_temp['ClusMembershipIDCol'] = refined_labels_col
                        ps_temp['PointsMinValue'] = 0
                        ps_temp['PointsMaxValue'] = ModelLabelsTotal - 1
                        print('Saving image of points coloured by updated classification label... ',end='', flush=True)
                        for ExportImage in ExportImageFormat:
                            if ModelLabelsTotal == 2:
                                fn_etc.make_image(datatable_clusIDs, os.path.join(outputpath_novel, Image_ScoresLabels_Folder, ps_current['FilePrefix'] + ' - Classification Labels (refined).' + ExportImage), ps_temp, cmap2ch)
                            else:
                                fn_etc.make_image(datatable_clusIDs, os.path.join(outputpath_novel, Image_ScoresLabels_Folder, ps_current['FilePrefix'] + ' - Classification Labels (refined).' + ExportImage), ps_temp, cmap3ch)
                        print('Done.')


                ### End of SaveMyImages

        print('Finished file ' + str(fileIdx + 1) + ' - ' + ps_current['FilePrefix'] + '\n')
    # End of per-fileIdx processing

    print('-------------------------------------------------\n' + \
                      '\t\t\tCompleted!' + \
          '\n-------------------------------------------------')
    print('The input folder was\t' + inputpath_novel)
    print('The output folder was\t' + outputpath_novel)
    if DoImage_ScoresHistogram:
        print('Histogram of model-score distribution were saved to folder \'' + Image_ScoresHistogram_Folder + '\'')
    if DoImage_ByModelScores or DoImage_ByLabels:
        print('Scatterplots for model scores and/or labels were saved to folder \'' + Image_ScoresLabels_Folder + '\'')
    if DoImage_InputValues:
        print('Input-value heatmaps were saved to folder \'' + Image_InputValues_Folder + '\'')
    
    if EmailUponCompletion:
        print(fn_etc.send_jobdone_email('Model evaluation (' + model_ID + ') on folder:\n' + outputpath_novel, 'Model evaluation done'))

