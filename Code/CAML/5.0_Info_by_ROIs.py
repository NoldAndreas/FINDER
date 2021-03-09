#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Takes data from previous stage (evaluation by model and cluster segmentation)
and extracts statistics

This will match a JSON formatted file of polygon coordinates (specifying the ROI shape) to a processed image file.

The name of the ROI file is important for it to be able to match to a processed image file

The name must be <name of the original file, less the file extension>_ROI_x.json

ROI_x where x is a number, indicating the roi number, ie you can have multiple ROIs applied to an image.

e.g. the original input image was:
    20190827-1_exciting_data.csv

and you have drawn two ROIs to examine, their names would be:
    20190827-1_exciting_data_ROI_1.json
    20190827-1_exciting_data_ROI_2.json

@author: dave

"""

import os
import numpy as np
from natsort import natsorted
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import pickle
import json
import seaborn as sns
from shapely.geometry import Polygon

proc_wd = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print('Changed working directory to ' + proc_wd)

import FuncEtc as fn_etc


# ========================================================================
# Email Notification
# ========================================================================
# Requires a valid email_secrets.json file: open email_secrets_DIST.json and save
# it as email_secrets.json. Fill out the settings with your own and save again.
DoEmailNotifications = 'ask' # 'yes' Emails will be sent when this stage ends.
                             # 'no'  Emails will not be sent.
                             # 'ask' Ask what to do for each run.


# ========================================================================
# Results export
# ========================================================================

DoAreaVsPointsPlot = True  # Save plot of cluster-area vs points-per-cluster. 
                           # Will be ignored if block_images is True (below)
AreaVsPointsPlot_folder = 'Area_v_Points' # Folder name for above images to be saved.

# data will be spat out as a pool with the data for all clusters in the ROI (or in
# the image if no ROIs) arranged in columns. If set to False then the data will be
# transposed and arranged in rows (one row per ROI)
export_pool_columns = True

summary_stats_fname = 'AAA - Summary Stats.csv'

area_factor = 1E6 # Areas will be in your original data's units, for example 'nm' 
                  # or '100 nm', squared. Use this factor to convert to SI units.
                  # If your data are in nm and you want nm² then set this to 1.0.
                  # To convert nm data to μm² set this to 1E6.
                  # To convert 100-nm ('pixel') data to μm² use 10 here.
                  
AreaUnits = 'um2' # This is the label for whatever units you decide to use.

# Summary data will be split up for easy import into Prism.
doPrismImportableFile = True
ForPrismFolder = 'For Prism'

# ========================================================================
# Image and plot settings
# ========================================================================
block_images = False     # overrides individual image settings to avoid saving 
                         # any images, anywhere.

ExportImageFormat = ['png'] # Format(s) to save images (given as a Python list).
                             # Format: 'eps', 'png', 'jpg', 'svg'
                             # Add multiple formats to the list and each image will
                             # be saved in that format. For example:                             
                             # ExportImageFormat = ['eps','png'] saves EPS and PNG.

#cmapfile = 'viridis'
cmapfile = './cmaps/candybright_256_rgb.txt'

# Light scheme, good for printing
plottxtcolor = (0,0,0)                  # text & line colour for all plots
plotbgcolor = (1,1,1)                   # background colour for all plots

## Dark scheme, good for presentations
#plottxtcolor = (1,1,1)                  # text & line colour for all plots
#plotbgcolor = (0.1765,0.1765,0.1765)    # background colour for all plots


# ========================================================================
# Queue Skipping
# ========================================================================
# Begin processing at this file in the list of files. Should normally be zero to
# start with the first file but you can jump ahead if you wish to resume processing
# an earlier set or if you are batch-processing across machines.
# NB: remember Python's zero-based indexing. To process the first file in the list
# this needs to be set to zero.
starting_index = 0   # normally zero (begin with the first file)

# End processing at this file index in the list of files. Should normally be zero
# to process all files in the list but you can terminate the list early, e.g.
# to only process a subset of the files or if you are batch-processing across 
# multiple machines
# NB: remember Python's zero-based indexing.
finishing_index = 0  # normally zero (end with the last file)

# NB starting_index can be greater than finishing_index; the script will just
# process the files in the reverse order. e.g.
#  FileIndices = [0,1,2,3,4,5,6,7,8,9]
#  starting_index = 0  and finishing_index = 5 >>> Processes Files 0,1,2,3,4
#  starting_index = 8  and finishing_index = 3 >>> Processes Files 8,7,6,5,4


# ========================================================================
#  End of user-editable variables
# ========================================================================

if __name__ == '__main__':

    cmap = fn_etc.load_colormap(cmapfile, flip=False)

    # if an outputpath from an earlier stage exists we can recycle it as an input
    # path suggestion
    if 'outputpath_novel' in locals():
        default_input_path = outputpath_novel
    elif 'inputpath_data' in locals():
        default_input_path = inputpath_data
    else:
        default_input_path = ''
        
    inputpath_data = fn_etc.askforinput(
            message = 'Folder with model-evaluated (\'Stage 4\') files (.pkl and .tsv)',
            errormessage= 'The folder you provided does not exist or you have supplied a file instead of a folder.',
            defaultval= default_input_path,
            isvalid = lambda v : os.path.isdir(v))

    ## get a list of the files to process from the given folder
    data_files_list = natsorted([i for i in os.listdir(inputpath_data) if 'ClusterShapes.pkl' in i])
    if np.shape(data_files_list)[0] > 0:
        total_files_novel = np.shape(data_files_list)[0]
    else:
        raise ValueError('No files to process! Check you have a) given the correct folder and b) this folder contains \'...ClusterShapes.pkl\' files from the \'Data preparation\' processing step.')

    # check the starting_index value in case we are restarting a run
    AskUseROIs = fn_etc.askforinput(
        message = 'Do you have ROIs to apply to your data (Y or N)',
        errormessage= 'Type Y or N',
        defaultval= 'Y',
        isvalid = lambda v : v.lower() in ['y','n','yes','no'])
    
    if AskUseROIs.lower() in ['y','yes']:
        doUseROIs = True
        
        if 'inputpath_rois' in locals():
            default_roi_input_path = inputpath_rois
        else:
            default_roi_input_path = ''
            
        inputpath_rois = fn_etc.askforinput(
                message = 'Folder with ROIs for each file (.json for each ROI)',
                errormessage= 'The folder you provided does not exist or you have supplied a file instead of a folder.',
                defaultval= default_roi_input_path,
                isvalid = lambda v : os.path.isdir(v))
    else:
        doUseROIs = False

    if doUseROIs:
        roi_files_novel = natsorted([i for i in os.listdir(inputpath_rois) if '.json' in i])
        total_rois = np.shape(roi_files_novel)[0]
        if total_rois > 0:

            #Trim the list of files to include only those with matching 
            keep_files = np.zeros(total_files_novel, dtype=bool)
            for test_idx, test_file in enumerate(data_files_list):
                test_file_base = os.path.splitext(test_file)[0].replace('_ClusterShapes','')
                test_rois_fnames = [roi for roi in roi_files_novel if test_file_base == roi.split('_roi')[0] or test_file_base == roi.split('_ROI')[0]]
                if len(test_rois_fnames) > 0:
                    keep_files[test_idx] = True
            data_files_list = [i for (i, v) in zip(data_files_list, keep_files) if v]
            total_files_novel_original = total_files_novel
            total_files_novel = np.shape(data_files_list)[0]
            if total_files_novel == 0:
                raise ValueError('Found ' + str(total_rois) + ' ROI(s) but could not match any ROI to a data file! Check your ROI names.')
            else:
                fn_etc.info_msg('Found ' + str(total_rois) + ' ROIs which apply to ' + str(total_files_novel) + ' files (of ' + str(total_files_novel_original) + ' files overall).')

        else:
            raise ValueError('There are no ROIs (files with .json extension) in the specified folder!')
    else:
        roi_files_novel = 'No ROIs will be used for this folder'
        total_rois = 1
        print('Found ' + str(total_files_novel) + ' datasets to work with...')

    # If we are exporting to Prism then we probably want to do some stats and we
    # probably have replicates in our dataset.
    if doPrismImportableFile:
        prism_split_replicants = fn_etc.askforinput(
            message = 'Prism Export: how many replications exist for each condition in this folder? (0 = disable)',
            errormessage= 'Please supply a positive integer value or supply 0 to disable Prism Export for this run.',
            defaultval= '1',
            isvalid = lambda v: v.isdigit() and int(v) >= 0)
        prism_split_replicants = int(prism_split_replicants)
        if prism_split_replicants == 0: doPrismImportableFile = False

    # specify an output folder
    # test for a stage4 folder name
    try:
        source_folder = os.path.split(inputpath_data)[1].split('_')
        if source_folder[:3] == ['4', 'evaluated', 'by']:
            eval_model = source_folder[3] + '_'
        else:
            eval_model = ''
    except:
        eval_model = ''
    if doUseROIs:
        ROI_root = os.path.basename(os.path.normpath(inputpath_rois))
    else:
        ROI_root = 'whole_image'
    outputpath_summary = fn_etc.askforinput(
            message = 'Output folder',
            errormessage= 'The dataset must be named for an output folder to be created!',
            defaultval= os.path.abspath(os.path.join(inputpath_data, '..', '5_ROIs_' + eval_model + ROI_root)),
            isvalid = lambda v : len(v) > 0)

    # check the starting_index value in case we are restarting a run
    if starting_index != 0:
        reset_starting_index = fn_etc.askforinput(
            message = 'Current Index is set to ' + str(starting_index) + '. Do you want to reset it to zero? (Y or N)',
            errormessage= 'Type Y or N',
            defaultval= 'Y',
            isvalid = lambda v : v.lower() in ['y','n','yes','no'])
        
        if reset_starting_index.lower() in ['y','yes']:
            starting_index = 0
    
    # check the finishing_index value in case we are restarting a run
    if finishing_index != 0:
        reset_finishing_index = fn_etc.askforinput(
            message = 'Current Index is set to ' + str(finishing_index) + ', i.e. end processing after File ' + str(finishing_index - 1) + ' is done. Do you want to reset it and process all files? (Y or N)',
            errormessage= 'Type Y or N',
            defaultval= 'y',
            isvalid = lambda v : v.lower() in ['y','n','yes','no'])
    
        if reset_finishing_index.lower() in ['y','yes']:
            finishing_index = total_files_novel
            print('Finishing Index has been reset and all files in the folder will be processed.')
        else:
            print('Keeping the current index. Processing will end once File ' + str(finishing_index) + ' is done.')
    else:
        finishing_index = total_files_novel

    if starting_index > total_files_novel or \
       finishing_index > total_files_novel or \
       finishing_index < starting_index:
           fn_etc.err_msg('The starting_index (set to ' + str(starting_index) + ') should be less than the finishing_index (set to ' + str(finishing_index) + ' and both should be within the range of files being processed (' + str(total_files_novel) + ' files matched with ROIs)')
           raise ValueError('Check that your starting and finishing points take account of the number of files to process (after excluding those without ROIs!)')


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
        

    proceed_with_processing = fn_etc.askforinput(
        message = 'When you are ready to proceed press Enter (or X to cancel everything and exit)',
        errormessage= 'Type P to proceed or X to exit',
        defaultval= 'P',
        isvalid = lambda v : v.lower() in ['p','x'])
    
    if proceed_with_processing.lower() in ['p']:
        print('Rightyo, off we go...')
    elif proceed_with_processing.lower() in ['x']:
        print('That\'s ok. Maybe next time?')
        raise ValueError('No errors, you just decided not to proceed and that\'s OK! :)')

    #make the folder for the output data
    if not os.path.exists(outputpath_summary):
        os.makedirs(outputpath_summary)

    if DoAreaVsPointsPlot and not os.path.exists(os.path.join(outputpath_summary, AreaVsPointsPlot_folder)):
        os.makedirs(os.path.join(outputpath_summary, AreaVsPointsPlot_folder))

    if doPrismImportableFile and not os.path.exists(os.path.join(outputpath_summary, ForPrismFolder)):
        os.makedirs(os.path.join(outputpath_summary, ForPrismFolder))
        
    # create a file to hold summary data as they come in
    summary_stats_fname = os.path.join(outputpath_summary, summary_stats_fname)
    
    with open(summary_stats_fname,'a') as fstats:
        writer = csv.writer(fstats)
        writer.writerow(['FileID', 'ROI No.', 'ROI Area (um2)', 'Points','Points per um2', 'Clustered Points','Clustered Points per um2', 'Percent Clustered', 'Clusters', 'Clusters per um2', 'Median Points per Cluster', 'Median Cluster Area (nm2)', 'Median Point Density', 'File Name'])
        
    stats_collector = np.zeros((total_files_novel * total_rois, 10)).astype(float)
    #        0 = fileIdx
    #        1 = ROI
    #        2 = TotalPointsThisImage
    #        3 = TotalClusteredPoints
    #        4 = PercentClustered
    #        5 = roi_total_clusters
    #        6 = Median_PtsPerCluster
    #        7 = Median_AreaPerCluster
    #        8 = Median_PtDensityPerCluster
    #        9 = Area of the ROI
    insertion_stats_row = 0

    # lists to hold pooled data from all ROIs in each image
    fname_pool = list()
    roi_pool = list()
    pts_per_cluster_pool = list()
    area_pool = list()
    density_pool = list()

    # some hard-coded data locations as our input PKL'd data doesn't hold all the other original input data
    xCol = 0
    yCol = 1
    ClusterIDCol = 2
    
    # ========================================================================
    #    For each file ... 
    # ========================================================================
    #        
    # using starting_index we can easily pick up again in case of stoppages
    for fileIdx in range(starting_index, finishing_index):
        
        starting_index = fileIdx # update starting index in case something goes wrong and we need to restart this loop
        current_file = data_files_list[fileIdx]

        if doUseROIs:
            # Find those ROI-files which match the current data-file (minus extensions and known suffixes!)
            current_file_base = os.path.splitext(current_file)[0].replace('_ClusterShapes','')
            my_rois_fnames = [roi for roi in roi_files_novel if current_file_base == roi.split('_roi')[0] or current_file_base == roi.split('_ROI')[0]]
            my_total_rois = len(my_rois_fnames)

        # load the data and extract the variables from the PKL
        with open(os.path.join(inputpath_data, current_file), 'rb') as f:
            recall_data = pickle.load(f, encoding='latin1')
    
        cluclub_path = recall_data[0]               # path outline
        cluclub_popn = np.array(recall_data[1])     # point within clusters
        cluclub_area = np.array(recall_data[2])     # area of clusters
        cluclub_cIDs = np.array(recall_data[3])     # list of cluster IDs
        data_essentials = recall_data[4]            # x, y, cluID
        ps_current = recall_data[5]                 # proc settings
        
        TotalPointsThisImage = data_essentials.shape[0]
        TotalClustersThisImage = cluclub_cIDs.shape[0]
        
        stats_fname = os.path.join(outputpath_summary, ps_current['FilePrefix'] + '_DataCalled.tsv')

        fn_etc.progress_msg(str(fileIdx + 1) + ' of ' + str(total_files_novel), ps_current['FilePrefix'])

        # Gather all the coords for all the ROIs from the ROI-files for this data-file
        my_rois = list()
        if doUseROIs:
            for roi_fname in my_rois_fnames:
                with open(os.path.join(inputpath_rois, roi_fname), 'r') as file:
                    new_roi = np.array(json.loads(file.read()))
                my_rois.append(new_roi)
                del(new_roi, roi_fname)
            print('Found ' + str(my_total_rois) + ' ROIs for this file.')
        else:
            # make a 'roi' covering the entire field of view
            full_field_roi = np.array([[ps_current['xMin'], ps_current['yMin']],
                                     [ps_current['xMin'], ps_current['yMax']],
                                     [ps_current['xMax'], ps_current['yMax']],
                                     [ps_current['xMax'], ps_current['yMin']],
                                     [ps_current['xMin'], ps_current['yMin']]])
            my_rois.append(full_field_roi)
            my_total_rois = 1
        
        if 'SaveMyImages' not in ps_current:
            ps_current['SaveMyImages'] = False
        
        # Get ready for image saving
        if ps_current['SaveMyImages']:
            
            # make a randomised colour LUT here, same length as the number of unique cluster IDs and with ID=0 as 'grey' for NC points.
            # we do this before looking at each ROI to keep colouring consistent between ROI images.
            vals = np.linspace(0, 1, TotalClustersThisImage + 1) # include NC points this time
            np.random.shuffle(vals)
            jumble_cmap = plt.cm.colors.ListedColormap(cmap(vals))
            jumble_cmap.set_under(color='grey') # colours outside the scale will be this color.
            
            ps_temp = ps_current
            if ps_temp['ImageSize'][0] / 50 < 300: # TODO: this needs to adapt to the actual ROI area rather than the original ImageSize
                ps_temp['OutputScaling'] = 1
                ps_temp['PointSize'] = 10
            else:
                ps_temp['OutputScaling'] = 100
                ps_temp['PointSize'] = 1.0
            ps_temp['PointsMinValue'] = np.min(cluclub_cIDs) + 1 # minimum needs to be wrt to the cluster IDs so that NC points become out of range and are coloured as 'under'.
            ps_temp['PointsMaxValue'] = np.max(cluclub_cIDs)
            ps_temp['xCol'] = xCol
            ps_temp['yCol'] = yCol
            ps_temp['ClusMembershipIDCol'] = ClusterIDCol
        
        # save per-cluster data for this ROI            
        cluclub_density = (cluclub_popn / cluclub_area)
        
#        # TODO - include cluster centroids here!
#        cluclub_centroids = 0

        for current_roi in range(my_total_rois):
            
            if doUseROIs:
                print('ROI ' + str(current_roi +1) + ' of ' + str(my_total_rois) + ':')
                roi_suffix = '_roi_' + str(current_roi + 1)
            else:
                print('File ' + str(fileIdx + 1) + ' - whole field, no ROIs:')
                roi_suffix =''
                
            current_roi_path = path.Path(my_rois[current_roi])
            current_roi_shape = Polygon(my_rois[current_roi])
            my_roi_area = current_roi_shape.area / area_factor
                    
            # points within ROI
            pts_in_roi = current_roi_path.contains_points(data_essentials[:,:ClusterIDCol])
            TotalPointsInRoi = sum(pts_in_roi)
            
            # get clusterIDs where at least one member-point lies within the ROI
            # This will find clusters which are on the ROI border as well as within it.
            # however the shape of the cluster outside the ROI
            clusterIDs_in_roi = np.unique(data_essentials[pts_in_roi, ClusterIDCol]).astype(int)
            clusterIDs_in_roi = clusterIDs_in_roi[clusterIDs_in_roi > 0] # exclude the zero IDs (NC points)
            cluster_idx_in_roi = clusterIDs_in_roi - 1 # convert ID into array indices
            roi_total_clusters = cluster_idx_in_roi.shape[0]

            # All clusters arising from points-witin-ROI
            clusterPaths_in_roi = [cluclub_path[patch_id] for patch_id in cluster_idx_in_roi[1:]]
            
            # Only those clusters which are entirely within the ROI
            clusterPaths_wholly_within_roi = [roi_patch for roi_patch in clusterPaths_in_roi if current_roi_path.contains_path(roi_patch)]
            
            # And those clusters which cross the ROI
            clusterPaths_on_edge_roi = [roi_patch for roi_patch in clusterPaths_in_roi if not current_roi_path.contains_path(roi_patch)]

            # do something with the crossing-clusters
            if len(clusterPaths_on_edge_roi) > 0:
                print('***\t' + str(len(clusterPaths_on_edge_roi)) + ' clusters found crossing the ROI boundary line')
                
#                    clusterPaths_snipped = list()
#                    clusterIDs_snipped = list()
#                    
#                    for border_cluster in clusterPaths_on_edge_roi:
#                        bc = Polygon(border_cluster.vertices)
#                        trimmed_c = bc.intersection(current_roi_shape)
#                        
#                        if trimmed_c.type == "MultiPolygon":
#                            for pgon in trimmed_c:
#                                pgon_path = path.Path(np.array(pgon.exterior.xy).T)
#                                
#                            #
#                        elif trimmed_c.type == "Polygon":
#                            trimmed_path = path.Path(np.array(trimmed_c.exterior.xy).T)
#                            clusterPaths_snipped.append(trimmed_path)

            # some stats about this thing
            #TotalClusteredPoints = np.sum(idx > 0 for idx in data_essentials[:, ClusterIDCol])
            TotalClusteredPointsInRoi = np.sum(np.fromiter((idx > 0 for idx in data_essentials[pts_in_roi, ClusterIDCol]), dtype=int))
            PercentClustered = (TotalClusteredPointsInRoi / TotalPointsInRoi) * 100
            print('\t' + str(TotalPointsInRoi) + ' points, ' + str(roi_total_clusters) + ' clusters, ' + str(round(PercentClustered, 2)) + ' percent of points in clusters.')

            # clusID, No. points it holds, clusArea, point-density
            per_cluster_stats = np.vstack((clusterIDs_in_roi, cluclub_popn[cluster_idx_in_roi], cluclub_area[cluster_idx_in_roi], cluclub_density[cluster_idx_in_roi])).T

            fname_pool.append(ps_current['FilePrefix'])
            roi_pool.append(roi_suffix)
            pts_per_cluster_pool.append(per_cluster_stats[:, 1])
            area_pool.append(per_cluster_stats[:, 2])
            density_pool.append(per_cluster_stats[:, 3])
            
            fileidx_clusstats_fname = os.path.join(outputpath_summary, ps_current['FilePrefix'] + roi_suffix + '_ClusterStats.csv')
            np.savetxt(fileidx_clusstats_fname, per_cluster_stats, '%i, %i, %.6f, %.6f', delimiter=',', header='ClusterID,Points,Area (' + AreaUnits + '),Density (pts/' + AreaUnits + ')', comments='')

            # add stats for this image to the collection
            Median_PtsPerCluster = np.median(per_cluster_stats[:, 1])     # Median Points per Cluster
            Median_AreaPerCluster = np.median(per_cluster_stats[:, 2])     # Median Cluster Area
            Median_PtDensityPerCluster = np.median(per_cluster_stats[:, 3])     # Median Density per Cluster
            
            # IQR_PtsPerCluster = np.subtract(*np.percentile(per_cluster_stats[:, 1], [75, 25]))
            
            stats_collector[insertion_stats_row,0] = fileIdx + 1
            stats_collector[insertion_stats_row,1] = current_roi + 1
            stats_collector[insertion_stats_row,2] = TotalPointsInRoi / my_roi_area
            stats_collector[insertion_stats_row,3] = TotalClusteredPointsInRoi / my_roi_area
            stats_collector[insertion_stats_row,4] = PercentClustered
            stats_collector[insertion_stats_row,5] = roi_total_clusters / my_roi_area
            stats_collector[insertion_stats_row,6] = Median_PtsPerCluster
            stats_collector[insertion_stats_row,7] = Median_AreaPerCluster
            stats_collector[insertion_stats_row,8] = Median_PtDensityPerCluster
            stats_collector[insertion_stats_row,9] = my_roi_area
    
            with open(summary_stats_fname,'a') as fstats:
                writer = csv.writer(fstats, delimiter=',')
                writer.writerow([fileIdx + 1, 
                                 current_roi + 1, 
                                 my_roi_area, 
                                 TotalPointsInRoi, 
                                 TotalPointsInRoi / my_roi_area, 
                                 TotalClusteredPointsInRoi, 
                                 TotalClusteredPointsInRoi / my_roi_area, 
                                 PercentClustered, 
                                 roi_total_clusters, 
                                 roi_total_clusters / my_roi_area, 
                                 Median_PtsPerCluster, 
                                 Median_AreaPerCluster, 
                                 Median_PtDensityPerCluster, 
                                 current_file])
            insertion_stats_row += 1

            if ps_current['SaveMyImages'] and not block_images:
                
                if doUseROIs:
                    
                    # set boundary to only include the area containing the ROI            
                    ROI_bounding_box = current_roi_path.get_extents().bounds
                    ps_temp['xMin'] = np.floor(ROI_bounding_box[0] / ps_temp['AutoAxesNearest']) * ps_temp['AutoAxesNearest']
                    ps_temp['xMax'] = np.ceil((ROI_bounding_box[2] + ROI_bounding_box[0])/ ps_temp['AutoAxesNearest']) * ps_temp['AutoAxesNearest']
                    ps_temp['yMin'] = np.floor(ROI_bounding_box[1] / ps_temp['AutoAxesNearest']) * ps_temp['AutoAxesNearest']
                    ps_temp['yMax'] = np.ceil((ROI_bounding_box[3] + ROI_bounding_box[1])/ ps_temp['AutoAxesNearest']) * ps_temp['AutoAxesNearest']
                    
                    ps_temp['ImageSize'][0] = np.min((ps_temp['xMax'] - ps_temp['xMin'], ps_temp['yMax'] - ps_temp['yMin']))
                    
                    test_dpi = ps_temp['ImageSize'][0] / 50
                    if test_dpi < 300:
                        ps_temp['OutputScaling'] = ps_temp['ImageSize'][0] / 300
                        ps_temp['PointSize'] = 5
                    else:
                        ps_temp['OutputScaling'] = 100
                        ps_temp['PointSize'] = 5

                    # All points with colour by ClusterID
                    print('\tSaving image of points coloured by cluster ID...', end='', flush=True)
                    for ExportImage in ExportImageFormat:
                        fn_etc.make_image(data_essentials[pts_in_roi,:], os.path.join(outputpath_summary, ps_current['FilePrefix'] + roi_suffix + '_ClusterAssignment.' + ExportImage), ps_temp, jumble_cmap)
                    print('Done.')
                    
                    print('\tSaving image of points with cluster outlines...', end='', flush=True)
                    plt.ioff()
                    fig = plt.figure(figsize=(30, 30))
                    ax = fig.add_subplot(111)
                    
                    # draw ROI shape
                    patch = patches.PathPatch(current_roi_path, facecolor='none', edgecolor='red')
                    ax.add_patch(patch)
            
                    # draw in-ROI cluster shapes
                    for in_roi_patch_idx, in_roi_patch_p in enumerate(clusterPaths_in_roi):
                        in_roi_patch = patches.PathPatch(in_roi_patch_p, facecolor=jumble_cmap(in_roi_patch_idx), edgecolor=None, alpha=0.5, lw=0.25, zorder=-1)
                        ax.add_patch(in_roi_patch)
                    
                    # draw in-ROI points
                    c_pts_in_roi = np.logical_and(data_essentials[:, ClusterIDCol] > 0, pts_in_roi)
                    nc_pts_in_roi = np.logical_and(data_essentials[:, ClusterIDCol] == 0, pts_in_roi)
                    plt.scatter(data_essentials[c_pts_in_roi, xCol], data_essentials[c_pts_in_roi, yCol], s=1.0, c='k', zorder=2, marker='.', edgecolors='none')
                    plt.scatter(data_essentials[nc_pts_in_roi, xCol], data_essentials[nc_pts_in_roi, yCol], s=1.0, c='grey', zorder=1, marker='.', edgecolors='none')
    
                    ax.set_xlim(ps_temp['xMin'],ps_temp['xMax'])
                    ax.set_ylim(ps_temp['yMin'],ps_temp['yMax'])
                    
                    cluster_outlines_fname = os.path.join(outputpath_summary, ps_current['FilePrefix'] + roi_suffix + ' - Cluster Outlines.')
                    
                else:
                    
                    print('\tSaving image of points with cluster outlines...', end='', flush=True)
                    plt.ioff()
                    fig = plt.figure(figsize=(30, 30))
                    ax = fig.add_subplot(111)
                    
                    # draw all cluster shapes
                    for patch_idx, patch_p in enumerate(cluclub_path):
                        patch = patches.PathPatch(patch_p, facecolor=jumble_cmap(patch_idx), edgecolor=None, alpha=0.5, lw=0.25, zorder=-1)
                        ax.add_patch(patch)
                    
                    # plot all points
                    c_pts_in_roi = data_essentials[:, ClusterIDCol] > 0
                    nc_pts_in_roi = data_essentials[:, ClusterIDCol] == 0
                    plt.scatter(data_essentials[c_pts_in_roi, xCol], data_essentials[c_pts_in_roi, yCol], s=0.5, c='k', zorder=2, marker='.', edgecolors='none')
                    plt.scatter(data_essentials[nc_pts_in_roi, xCol], data_essentials[nc_pts_in_roi, yCol], s=0.5, c='grey', zorder=1, marker='.', edgecolors='none')
                    
                    # set boundaries to field of view
                    ax.set_xlim(ps_current['xMin'],ps_current['yMax'])
                    ax.set_ylim(ps_current['yMin'],ps_current['yMax'])
                    
                    cluster_outlines_fname = os.path.join(outputpath_summary, ps_current['FilePrefix'] + ' - Cluster Outlines.')
                    
                plt.rcParams['figure.figsize'] = [30, 30]
                ax.set_aspect('equal')
                
                test_dpi = ps_temp['ImageSize'][0] / 50
                if test_dpi < 300:
                    use_dpi = 300
                else:
                    use_dpi = test_dpi
                
                for ExportImage in ExportImageFormat:
                    exportname = cluster_outlines_fname + ExportImage
                    fig.savefig(exportname,
                                dpi=use_dpi,
                                bbox_inches=0,
                                facecolor=fig.get_facecolor(),
                                edgecolor='none',
                                transparent=True)
                    
                plt.close('all')        
                plt.ion() # turn on interactive mode
                print('Done.')
            ### End of SaveMyImages

            if DoAreaVsPointsPlot and not block_images:
                print('\tSaving Cluster-Area vs Points-per-Cluster histogram...', end='', flush=True)
                
                if doUseROIs:
                    dual_plot_fname = os.path.join(outputpath_summary, AreaVsPointsPlot_folder, ps_current['FilePrefix'] + roi_suffix + ' - Points vs Area')
                else:
                    dual_plot_fname = os.path.join(outputpath_summary, AreaVsPointsPlot_folder, ps_current['FilePrefix'] + ' - Points vs Area')
                
                # Area-per-Cluster vs Points-per-Cluster
                
                # Points
                min_x = 3 # cluster count minimum # np.floor(np.min(per_cluster_stats[:, 1]))
                max_x = np.ceil(np.max(per_cluster_stats[:, 1]))
                
                # Area
                min_y = 10 # nm2 area minimum # np.floor(np.min(per_cluster_stats[:, 2]))
                max_y = np.ceil(np.max(per_cluster_stats[:, 2]))
                
                plt.ioff()
                
#                # linear-linear plot
#                mybins_x = range(3, int(max_x), 1)
#                mybins_y = range(0, int(np.ceil(max_y / 1000) * 1000), 100)
#                g = sns.JointGrid(per_cluster_stats[:, 1],
#                                  per_cluster_stats[:, 2],
#                                  xlim=[min_x, max_x],
#                                  ylim=[min_y, max_y]
#                                 )
#                
#                _ = g.ax_marg_x.hist(per_cluster_stats[:, 1], color="k", alpha=0.6, bins=mybins_x)
#                _ = g.ax_marg_y.hist(per_cluster_stats[:, 2], color="k", alpha=0.6, bins=mybins_y, orientation="horizontal")
#                
#                g.plot_joint(plt.scatter, color='black', edgecolor=None, marker='.', s=1)
#                g.plot_joint(sns.kdeplot, n_levels=20, shade=False, gridsize=100, cmap="plasma")
#    
#                ax = g.ax_joint
#                ax.set(xlabel='Points per cluster', ylabel='Area per cluster (nm²)')
#                # sns.despine(offset=5, trim=False)
#                # plt.tight_layout()
#    
#                g.savefig(dual_plot_fname + ' (linear).png')
#                g.savefig(dual_plot_fname + ' (linear).svg')
#                plt.close('all')
                
                # log-log plot
                mybins_x = range(3, int(max_x), 1) 
                mybins_y_log = np.logspace(0, np.ceil(np.log10(max_y)), 100)
                g = sns.JointGrid(per_cluster_stats[:, 1],
                                  per_cluster_stats[:, 2],
                                  xlim=[min_x, max_x],
                                  ylim=[min_y, max_y],
                                 )
                
                _ = g.ax_marg_x.hist(per_cluster_stats[:, 1], color="k", alpha=0.6, bins=mybins_x)
                _ = g.ax_marg_y.hist(per_cluster_stats[:, 2], color="k", alpha=0.6, bins=mybins_y_log, orientation="horizontal")
                g.ax_marg_x.set_xscale('log')
                g.ax_marg_y.set_yscale('log')
                
                g.plot_joint(plt.scatter, color='black', edgecolor=None, marker='.', s=1)
                g.plot_joint(sns.kdeplot, n_levels=20, shade=False, gridsize=100, cmap="plasma")
    
                ax = g.ax_joint
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set(xlabel='Points per cluster', ylabel='Area per cluster (nm²)')
                # sns.despine(offset=5, trim=False)
                # plt.tight_layout()
                
                g.savefig(dual_plot_fname + ' (log-log).png')
                g.savefig(dual_plot_fname + ' (log-log).svg')
                plt.close('all')
                
                plt.ion()
                print('Done.')
            ### End of DoAreaVsPointsPlot

        ### End of per-ROI loop

    ### End of per-File loop

    ### Dump the pooled data
    max_length = len(sorted(pts_per_cluster_pool,key=len, reverse=True)[0])
    max_file_rois = len(fname_pool)
    
    if export_pool_columns:
        
        # pools are arranged as one column per file-roi
        # this makes it easier to import the pooled data into prism as a Column Table
        PtsPerCluster_pzf = np.full_like(np.empty((max_length, max_file_rois)), np.nan)
        AreaPerCluster_pzf = np.full_like(np.empty((max_length, max_file_rois)), np.nan)
        DensityPerCluster_pzf = np.full_like(np.empty((max_length, max_file_rois)), np.nan)
        
        pool_header=''
        for i in range(max_file_rois):
            PtsPerCluster_pzf[:len(pts_per_cluster_pool[i]), i] = pts_per_cluster_pool[i]
            AreaPerCluster_pzf[:len(area_pool[i]), i] = area_pool[i]
            DensityPerCluster_pzf[:len(density_pool[i]), i] = density_pool[i]
            
            # build the column-header
            pool_header += ',' + fname_pool[i] + roi_pool[i]
        
        pool_header = pool_header[1:] # remove the leading delimiter character
        
        # points per cluster
        pool_fname_save = os.path.join(outputpath_summary, ForPrismFolder, 'points-per-cluster_pool_by_cols.csv')
        np.savetxt(pool_fname_save, PtsPerCluster_pzf, '%.0f', delimiter=',', header=pool_header, comments='')
        
        # area (per cluster)
        pool_fname_save = os.path.join(outputpath_summary, ForPrismFolder, 'area-per-cluster_pool_by_cols.csv')
        np.savetxt(pool_fname_save, AreaPerCluster_pzf, '%.6f', delimiter=',', header=pool_header, comments='')
        
        # density (per cluster)
        pool_fname_save = os.path.join(outputpath_summary, ForPrismFolder, 'density-per-cluster_pool_by_cols.csv')
        np.savetxt(pool_fname_save, DensityPerCluster_pzf, '%.6f', delimiter=',', header=pool_header, comments='')
        
    else:

        # pools are arranged as one row per file-roi
        PtsPerCluster_pzf = np.full_like(np.empty((max_file_rois, max_length)), np.nan)
        AreaPerCluster_pzf = np.full_like(np.empty((max_file_rois, max_length)), np.nan)
        DensityPerCluster_pzf = np.full_like(np.empty((max_file_rois, max_length)), np.nan)
        
        for i in range(max_file_rois):
            PtsPerCluster_pzf[i,:len(pts_per_cluster_pool[i])] = pts_per_cluster_pool[i]
            AreaPerCluster_pzf[i,:len(area_pool[i])] = area_pool[i]
            DensityPerCluster_pzf[i,:len(density_pool[i])] = density_pool[i]

        # build the row-header
        rows = [m+n for m,n in zip(fname_pool, roi_pool)]
        row_headers = np.array(rows)[:, np.newaxis]
            
        # points per cluster
        pool_fname_save = os.path.join(outputpath_summary, ForPrismFolder, 'points-per-cluster_pool_by_rows.csv')
        str_data = np.char.mod("%.0f", PtsPerCluster_pzf)
        np.savetxt(pool_fname_save, np.hstack((row_headers, str_data)), delimiter=', ', fmt='%s')
        
        # area (per cluster)
        pool_fname_save = os.path.join(outputpath_summary, ForPrismFolder, 'area-per-cluster_pool_by_rows.csv')
        str_data = np.char.mod("%10.6f", AreaPerCluster_pzf)
        np.savetxt(pool_fname_save, np.hstack((row_headers, str_data)), delimiter=', ', fmt='%s')
        
        # density (per cluster)
        pool_fname_save = os.path.join(outputpath_summary, ForPrismFolder, 'density-per-cluster_pool_by_rows.csv')
        str_data = np.char.mod("%10.6f", DensityPerCluster_pzf)
        np.savetxt(pool_fname_save, np.hstack((row_headers, str_data)), delimiter=', ', fmt='%s')
    
    # Pool-of-eeeeeverything
    OverAllDump = np.empty((np.sum(~np.isnan(PtsPerCluster_pzf)),3)) * np.nan
    OverAllDump[:,0] = PtsPerCluster_pzf[~np.isnan(PtsPerCluster_pzf)]
    OverAllDump[:,1] = AreaPerCluster_pzf[~np.isnan(AreaPerCluster_pzf)]
    OverAllDump[:,2] = DensityPerCluster_pzf[~np.isnan(DensityPerCluster_pzf)]
    OverAllDump_headers = 'Points per Cluster,Area per Cluster, Density per Cluster'
    overalldump_fname = os.path.join(outputpath_summary, ForPrismFolder, 'ALL_clusters_pooled.csv')
    np.savetxt(overalldump_fname, OverAllDump, '%.6f', delimiter=',', header=OverAllDump_headers, comments='')
    
    ### end of pool dump

    # Rearrange some data for easier importing into Graphpad Prism
    if doPrismImportableFile:
    
        prism_split_rows = np.shape(stats_collector)[0]
        prism_split_columns = np.shape(stats_collector)[1]
        prism_split_conditions = int(prism_split_rows / prism_split_replicants)
       
        column_titles = ['fileIDx', 'RoI', 'TotalPoints', 'TotalClustered Points', 'PercentPointsClustered', 'TotalClusters', 'MedianPointsPerCluster', 'MedianAreaPerCluster', 'MedianPointDensityPerCluster', 'ROI Area']
    
        repeats =  np.tile(np.array(range(prism_split_replicants)), (prism_split_conditions, 1)).reshape((prism_split_rows,1), order='F')
        
        print('Exporting for Prism:')
        
        for prism_col in range(prism_split_columns):
    
            filename_base = column_titles[prism_col]
    
            print('\t - ' + filename_base)
    
            tmp_data = np.zeros((prism_split_replicants, prism_split_conditions))
    
            insert_row = 0
            for start_rep in range(prism_split_replicants):
                keep_repeats = repeats == start_rep
                tmp_data[insert_row, :] = stats_collector[keep_repeats[:,0], prism_col].T
                insert_row += 1
                
            for build_header in range(prism_split_conditions):
                if build_header == 0 :                
                    out_header = 'File1'
                else:
                    out_header = out_header + ',File' + str(build_header + 1)
                build_header += 1
        
            filename_save = os.path.join(outputpath_summary, ForPrismFolder, filename_base + '_forPrism.csv')
            np.savetxt(filename_save, tmp_data, '%.6f', delimiter=',', header=out_header, comments='')
    # End of export for Prism

    print('-------------------------------------------------\n' + \
                      '\t\t\tCompleted!' + \
          '\n-------------------------------------------------')
    print('The input folder was\t' + inputpath_data)
    print('The output folder was\t' + outputpath_summary)
    if EmailUponCompletion:
        CompletionMsg = 'Extract cluster information from ROIs\r\n\r\nData folder:\r\n' + inputpath_data + '\r\n\r\nROI folder:\r\n'
        if doUseROIs:
            CompletionMsg += inputpath_rois
        else:
            CompletionMsg += 'None (whole cell examined)'
        print(fn_etc.send_jobdone_email(CompletionMsg, 'Info from ROIs done'))

