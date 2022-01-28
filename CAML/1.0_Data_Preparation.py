#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepares SMLM data for subsquent stages.

This script takes SMLM data tables as input and for all points measures their
distances to their nearest neighbours. These are saved as memory-mapped binary
files for the subsequent stages.

@author: Dave

"""

import os
import numpy as np
from natsort import natsorted
import gc
import time
import multiprocessing
from joblib import Parallel, delayed
import csv
import datetime
import json

proc_wd = os.path.dirname(os.path.abspath(__file__))
print(proc_wd);

if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print('Changed working directory to ' + proc_wd)

from CAML import FuncEtc as fn_etc
from CAML import FuncDistCalcs as fn_distcalc
from CAML import FuncDistCalcs_3D as fn_distcalc_3D

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
SaveImages = False                  # Save images (scatterplots) of the data?

ExportImageFormat = ['png'] # Format(s) to save images (given as a Python list).
                             # Format: 'eps', 'png', 'jpg', 'svg'
                             # Add multiple formats to the list and each image will
                             # be saved in that format. For example:                             
                             # ExportImageFormat = ['eps','png'] saves EPS and PNG.

# For simulated data, you can save an image for every dataset ('all') or choose to
# save an image for only one repeat within each cluster scenario ('partial').
# This setting requires input images to have specific patterns in their filenames
# which will be present in the simulated files.
SimuCell_SaveImagesFor = 'partial' # 'all'     Save all repeats in a scenario group
                                   # 'partial' Save 1 repeat from each scenario
                                   # If SaveImage is False (above) then no images
                                   # for simulated data will be saved.

# The decision to save an image here will be transmitted to the later stages of
# processing.

# ========================================================================
# Sanity Checking
# ========================================================================
# It's easy to get things mixed up so we can check for that before we rush
# headlong into processing the wrong columns or whatever
doSanityCheck = True
# Specify your sensible ranges for the xy (and z) data below.
# Generally SMLM data has a fairly squarish aspect ratio unless you're one of these
# sadists who likes taking weirdly shaped camera ROIs
Sane_min_xy_ratio =  0.1  # expect x value range to be larger than this multiple of the y value range
Sane_max_xy_ratio = 10.0  # expect x value range to be smaller than this multiple of the y value range
Sane_max_zx_ratio =  2.0  # expect z value range to be smaller than this multiple of the x value range
Sane_max_zy_ratio =  2.0  # expect z value range to be smaller than this multiple of the y value range
SanityCheck_FirstOnly = False # only do this check for the first file.
# Sanity means a vaguely square aspect ratio (from 1:1 to 10:1 by any axis) as
# sometimes giving the wrong xy columns doesn't break processing flow but does
# mean the output is confusing and meaningless.
# Sanity also means all points must be located within the stated boundaries 
# although bear in mind there are exceptions e.g. for huge drift correction or 
# multichannel offsets.

# Screen for identical xy coordinate pairs
# If found only the first pair will be retained and the other removed.
# Identical points give rise to zero-length neighbour distances which can break 
# things further down the line.
doFilterDuplicates = True 


# ========================================================================
# Isometric Grid (not finished or implemented yet)
# ========================================================================
# Add isometrically distributed points prior to distance measurement
# These points will be removed after the model-assessment stage but can help
# in cases of very high degree of clustering.
doAppendIsoGrid = False # This isn't really finished yet and should be left as false.


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
# Parallel Processing
# ========================================================================
# How many of the system's processors to devote to processing?
# You will want to keep 1 or 2 CPUs free for background tasks to run.
total_cpus = multiprocessing.cpu_count() - 2
# N.B.: If you want to disable parallel processing, set total_cpus = 1

# Processing time estimation.
# The times given for processing a data table here are from my machine, using 
# 10 cores of a Core i7-4930K CPU at 3.40 GHz and Debian Stretch.
# If you have evaluated your own  machine on various sizes of input data, you 
# should be able to fit a quadratic function as a rough estimator of the processing
# time (which is what is used below to give time estimates).
# You can edit your own values to get a better estimage of the times:
# ProcTime = (ProcTimeEst_A * Points^2) + (ProcTimeEst_B * Points) + ProcTimeEst_C
# ProcTime is processing time in seconds, Points is total points in a data file.
# ProcTime includes time taken to load the data, calculate and save distances, and
# save high-resolution preview images.
ProcTimeEst_A = 4.61E-9
ProcTimeEst_B = -2.92E-4
ProcTimeEst_C = 0.0


# ========================================================================
#  End of user-editable variables
# ========================================================================

if True:#__name__ == '__main__':
    
    # Initial processing settings (ProcSettings) are loaded from a JSON file
    good_json = False
    default_json_file = ''
    
    while not good_json:

        if 'input_PrepJSON' in locals():
            default_json_file = input_PrepJSON # from a previous run of this script
        elif 'zzz_json_fname' in locals():
            default_json_file = zzz_json_fname # from zzz_MakeJSON.py

        # input_PrepJSON = fn_etc.askforinput(
        #     message = 'Full path to JSON file describing the data',
        #     errormessage = 'The file you provided does not exist, or is not a \'.json\' file, or you supplied a folder path. Please supply the full path to a \'.json\' file.',
        #     defaultval = default_json_file,
        #     isvalid = lambda v : os.path.isfile(v) and os.path.splitext(v)[1]=='.json')
        input_PrepJSON = default_json_file;
        
        with open(input_PrepJSON, 'r') as file:
            ps = json.loads(file.read())
        
        # check here that the ImageSize is valid. Older versions used a single int
        # and assumed a square 2D field. We can convert those older values here.
        if type(ps['ImageSize']) == int:
            ps['ImageSize'] = [ps['ImageSize'], ps['ImageSize'], 0]
        
        # we'll need to know if we are using 2D or 3D images
        if ps['ImageSize'][2] == 0:
            ps['three_dee'] = False
        else:
            ps['three_dee'] = True
        
        fn_etc.info_msg('Imported JSON variables:')
        print(' │')
        print(' ├─InputFileDelimiter:\t' + ps['InputFileDelimiter'])
        print(' ├─InputFileExt:\t' + ps['InputFileExt'])
        print(' │')
        print(' ├─xCol:\t\t' + str(ps['xCol']))
        print(' ├─yCol:\t\t' + str(ps['yCol']))
        if ps['three_dee']:
            print(' ├─zCol:\t\t' + str(ps['zCol']))
        print(' ├─ClusMembershipIDCol:\t' + str(ps['ClusMembershipIDCol']))
        print(' ├─ChanIDCol:\t\t' + str(ps['ChanIDCol']))
        print(' ├─UIDCol:\t\t' + str(ps['UIDCol']))
        if 'LabelIDCol' in ps:
            print(' ├─LabelIDCol:\t\t' + str(ps['LabelIDCol']))
        print(' │')
        print(' ├─AutoAxes:\t\t' + str(ps['AutoAxes']))
        if ps['AutoAxes']:
            print(' ├─AutoAxesNearest:\t' + str(ps['AutoAxesNearest']))
            print(' ├─ImageSize:\t\tTo be determined')
            print(' ├─xMin:\t\tTo be determined')
            print(' ├─xMax:\t\tTo be determined')
            print(' ├─yMin:\t\tTo be determined')
            print(' ├─yMax:\t\tTo be determined')
            if ps['three_dee']:
                print(' ├─zMax:\t\tTo be determined')
                print(' ├─zMax:\t\tTo be determined')
        else:
            print(' ├─AutoAxesNearest:\tNot applicable')
            print(' ├─ImageSize:\t\t' + str(ps['ImageSize']))
            print(' ├─xMin:\t\t' + str(ps['xMin']))
            print(' ├─xMax:\t\t' + str(ps['xMax']))
            print(' ├─yMin:\t\t' + str(ps['yMin']))
            print(' ├─yMax:\t\t' + str(ps['yMax']))
            if ps['three_dee']:
                print(' ├─zMax:\t\t' + str(ps['zMin']))
                print(' ├─zMax:\t\t' + str(ps['zMax']))
        print(' │')
        print(' ├─ClosestFriend:\t' + str(ps['ClosestFriend']))
        print(' └─FurthestFriend:\t' + str(ps['FurthestFriend']))

        # verify_good_json = fn_etc.askforinput(
        #         message = 'Are these settings correct? Enter \'Y\' to proceed or enter \'N\' to select another JSON file (or the same file, after you have edited it with the correct settings)',
        #         errormessage= 'Type Y or N',
        #         defaultval= 'y',
        #         isvalid = lambda v : v.lower() in ['y','n','yes','no'])
        verify_good_json = 'y';       
        
        if verify_good_json.lower() in ['y','yes']:
            print('JSON file accepted.')
            good_json = True
        else:
            print('JSON file rejected.')

    if SaveImages and (SimuCell_SaveImagesFor == 'all' or SimuCell_SaveImagesFor == 'partial'):
        cmap2ch = fn_etc.load_colormap('./cmaps/two_channel_bluyel.txt', flip=False)
    
    # we will add a UID for each point in a file so we can track things across arrays.
    if not ps['UIDCol']:
        doAddUID = True
    else:
        doAddUID = False
    
    # If your data are not consistently located within the same field of view then the
    # image axes can be adjusted to accomodate based on the range of each 
    # image's xy data.
    if ps['AutoAxes']:
        doSetUpAxes = True
    else:
        doSetUpAxes = False

    default_input_path = os.path.dirname(input_PrepJSON)
    
    # get the data from the folder
    # inputpath = fn_etc.askforinput(
    #         message = 'Enter the path of the folder containing ' + ps['InputFileExt'] + ' data tables',
    #         errormessage= 'The folder you provided does not exist or you have provided the path to a file.',
    #         defaultval= default_input_path,
    #         isvalid = lambda v : os.path.isdir(v))
    inputpath = default_input_path;
    
    # ask_prepType = fn_etc.askforinput(
    #     message = 'Are these data to be evaluated as [1]-Training or [2]-Novel data? (Enter 1 or 2)',
    #     errormessage= 'Type the number 1 or 2 and press enter',
    #     defaultval= '2',
    #     isvalid = lambda v : v in ['1','2'])
    ask_prepType = '2';
        
    if ask_prepType in ['1']:
        if ps['ClusMembershipIDCol'] == False or ps['LabelIDCol'] == False:
            fn_etc.err_msg('You have not specified ClusMembershipIDCol or LabelIDCol in ProcSettings. Can\t train without knowing where the labels are!')
            raise ValueError('A value for ClusMembershipIDCol and LabelIDCol in ProcSettings is required when training. Please specify one.')
        else:
            prepType = 'training'
    elif ask_prepType in ['2']:
        prepType = 'novel'
        ps['ClusMembershipIDCol'] = False
        ps['LabelIDCol'] = False
    
    default_outfolder = os.path.join(output_folder, '1_prep_dNN(' + str(ps['ClosestFriend']) + '-' + str(ps['FurthestFriend']) + ')_' + prepType)
    # s1_prep_outputpath = fn_etc.askforinput(
    #         message = 'Enter the name of the output folder',
    #         errormessage= 'The output folder must be named',
    #         defaultval= default_outfolder,
    #         isvalid = lambda v : len(v) > 0 and not v.isspace())
    s1_prep_outputpath = default_outfolder;
   
    # get a list of input files from the given inputfolder
    files = natsorted([i for i in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, i)) and ps['InputFileExt'] in i])
    total_files = np.shape(files)[0]

    # check the starting_index value in case we are restarting a run
    if starting_index != 0:
        reset_starting_index = fn_etc.askforinput(
            message = 'Current Index is set to ' + str(starting_index) + ', i.e. begin with File ' + str(starting_index + 1) + '. Do you want to reset it to zero? (Y or N)',
            errormessage= 'Type Y or N',
            defaultval= 'y',
            isvalid = lambda v : v.lower() in ['y','n','yes','no'])
    
        if reset_starting_index.lower() in ['y','yes']:
            starting_index = 0
            print('Current index has been reset to zero. Processing will begin from the first file in the list.')
        else:
            print('Keeping the current index. Processing will begin with File ' + str(starting_index + 1) + ' in the list.')
    current_index = starting_index


    # check the finishing_index value in case we are restarting a run
    if finishing_index != 0:
        reset_finishing_index = fn_etc.askforinput(
            message = 'Current Index is set to ' + str(finishing_index) + ', i.e. end processing after File ' + str(finishing_index - 1) + ' is done. Do you want to reset it and process all files? (Y or N)',
            errormessage= 'Type Y or N',
            defaultval= 'y',
            isvalid = lambda v : v.lower() in ['y','n','yes','no'])
    
        if reset_finishing_index.lower() in ['y','yes']:
            finishing_index = total_files
            print('Finishing Index has been reset and all files in the folder will be processed.')
        else:
            print('Keeping the current index. Processing will end once File ' + str(finishing_index) + ' is done.')
    else:
        finishing_index = total_files


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
        print('Rightyo, off we go...')
        if total_cpus > 1 and os.name == 'posix':
            multiprocessing.set_start_method('forkserver', force=True)
    elif proceed_with_processing.lower() in ['x']:
        print('That\'s ok. Maybe next time?')
        raise ValueError('Not an error, you just decided not to proceed and that\'s OK! :)')

    #make the folder for the output data
    if not os.path.exists(s1_prep_outputpath):
        os.makedirs(s1_prep_outputpath)
        
    # open a log file for the processing
    # If you start processing part-way through the list of files you will have a 
    # new header line inserted to indicate where you picked up from.
    with open(os.path.join(s1_prep_outputpath, 'PrepLog.txt'), 'a+', newline='') as prep_log:
        writer = csv.writer(prep_log, delimiter='\t')
        if prepType == 'training':
            writer.writerow(['FileID', 'Total Points', 'ProcTime (s)', 'Clustered Points', 'Non-Clustered Points', 'Total Clusters', 'Excl. Duplicate Pts'])
        elif prepType == 'novel':
            writer.writerow(['FileID', 'Total Points', 'ProcTime (s)', 'Excl. Duplicate Pts'])
    
    # process all the files
    for fileIdx in range(starting_index, finishing_index):

        current_file = files[fileIdx]
        output_prefix = os.path.splitext(current_file)[0]
        fn_etc.info_msg(str(fileIdx + 1) + ' of ' + str(total_files) + '\t' + current_file)
        print('Loading data...', end='', flush=True)
        datatable = np.genfromtxt(os.path.join(output_folder, current_file),
                                  delimiter=ps['InputFileDelimiter'],
                                  skip_header=1) # names=True
        # usecols=(ps['xCol'],ps['yCol']) is an option here but ultimately we want to regenerate the full data table with the addition of clustering info columns

        # scale for data which is in daft units
        if 'DataScale' in ps:
            datatable[:,ps['xCol']] = ps['DataScale'] * datatable[:,ps['xCol']]
            datatable[:,ps['yCol']] = ps['DataScale'] * datatable[:,ps['yCol']]
        
            if ps['three_dee']:
                datatable[:,ps['zCol']] = ps['DataScale'] * datatable[:,ps['zCol']]

#        # will be exporting as comma-delimited from here, so swap out original delimiters in the header for commas
#        with open(os.path.join(inputpath, current_file), 'r') as f:
#            ps['TableHeaders'] = f.readline().strip()
#            if ps['InputFileDelimiter'] != ',':
#                ps['TableHeaders'] = ps['TableHeaders'].replace(ps['InputFileDelimiter'], ',')

        # retrieve headers from the data table
        with open(os.path.join(output_folder, current_file), 'r') as f:
            ps['TableHeaders'] = f.readline().strip()
        ps['TableHeaders'] = ps['TableHeaders'].replace('"', '') # clean up headers for legibility
    
        TotalPointsThisImage = datatable.shape[0]
        print('Done (' + str(TotalPointsThisImage) + ' points)')

        # warn about insufficient points in this image
        if TotalPointsThisImage < ps['FurthestFriend'] - ps['ClosestFriend']:
            fn_etc.warn_msg('This image has ' + str(TotalPointsThisImage) + ' points. Minimum of ' + str(ps['FurthestFriend'] - ps['ClosestFriend'] + 1) + ' points is required, according to the supplied JSON file. Missing neighbours will be padded with nans!')
            
        # auto image boundaries; avoid plotting cropped regions on a full-sized field.
        if doSetUpAxes == True:
            
            # get xy range of the data
            xmin = np.min(datatable[:,ps['xCol']])
            xmax = np.max(datatable[:,ps['xCol']])
            ymin = np.min(datatable[:,ps['yCol']])
            ymax = np.max(datatable[:,ps['yCol']])
            if ps['three_dee']:
                zmin = np.min(datatable[:,ps['zCol']])
                zmax = np.max(datatable[:,ps['zCol']])

    #        if xmin < 0:
    #            datatable[:,ps['xCol']] += abs(xmin)
    #            
    #        if ymin < 0:
    #            datatable[:,ps['yCol']] += abs(ymin)
            
            ps['xMin'] = np.floor(xmin / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
            ps['xMax'] = np.ceil(xmax / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
            ps['yMin'] = np.floor(ymin / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
            ps['yMax'] = np.ceil(ymax / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
            if ps['three_dee']:
                ps['zMin'] = np.floor(zmin / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
                ps['zMax'] = np.ceil(zmax / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
                ps['ImageSize'] = [ps['xMax'] - ps['xMin'], ps['yMax'] - ps['yMin'], ps['zMax'] - ps['zMin']]
                print('AutoAxes: \t Set image boundaries to [ ' + str(ps['xMin']) + ', ' + str(ps['xMax']) + ', ' + str(ps['yMin']) + ', ' + str(ps['yMax']) + ', ' + str(ps['zMin']) + ', ' + str(ps['zMax']) + ' ]')
            else:
                ps['ImageSize'] = [ps['xMax'] - ps['xMin'], ps['yMax'] - ps['yMin'],0]
                print('AutoAxes: \t Set image boundaries to [ ' + str(ps['xMin']) + ', ' + str(ps['xMax']) + ', ' + str(ps['yMin']) + ', ' + str(ps['yMax']) + ' ]')
            ps['AutoAxes'] = False
            

        #
        # Start of Sanity Check
        #    
        # 'Sane data' features:
        #   - square-ish aspect ratio (at least 1:10 or squarer). Wrong column 
        #     identifiers will usually have some insane aspect ratio.
        #   - all points are within the image boundaries give by the 
        #     'ps_input' list.
        # 
        # If your data are routinely failing these checks (e.g. you are not
        # using SMLM data as input) then you can disable the sanity checks at
        # the top of this script.
        # 
        if doSanityCheck:
            
            print('SanityCheck:\tX data in column ' + str(ps['xCol']) )
            
            # Check for invalid numbers in x col.
            if np.isnan(datatable[:,ps['xCol']]).any():
                fn_etc.err_msg('NaNs found in your input data x-coord column.')
                raise ValueError('Coordinate xy data must be real numbers, not NaNs.')

            print('SanityCheck:\tY data in column ' + str(ps['yCol']) )
            
            # Check for invalid numbers in y col.
            if np.isnan(datatable[:,ps['yCol']]).any():
                fn_etc.err_msg('NaNs found in your input data y-coord column.')
                raise ValueError('Coordinate xy data must be real numbers, not NaNs.')
            
            xy_range = [np.floor(min(datatable[:, ps['xCol']])),
                        np.ceil(max(datatable[:, ps['xCol']])), 
                        np.floor(min(datatable[:, ps['yCol']])),
                        np.ceil(max(datatable[:, ps['yCol']]))
                        ]
            
            if ps['three_dee']:
                print('SanityCheck:\tZ data in column ' + str(ps['zCol']) )
                
                # Check for invalid numbers in z col.
                if np.isnan(datatable[:,ps['zCol']]).any():
                    fn_etc.err_msg('NaNs found in your input data z-coord column.')
                    raise ValueError('Coordinate z data must be real numbers, not NaNs.')
                
                z_range = [np.floor(min(datatable[:, ps['zCol']])), 
                           np.ceil(max(datatable[:, ps['zCol']]))
                           ]
                print('Data ranges:\t' + str(xy_range[1] - xy_range[0]) + '(x) ' + 
                      str(xy_range[3] - xy_range[2]) + '(y) ' + 
                      str(z_range[1] - z_range[0]) + '(z)')
            else:
                print('Data ranges:\t' + str(xy_range[1] - xy_range[0]) + '(x) ' + 
                      str(xy_range[3] - xy_range[2]) + '(y)')

            print('SanityCheck:\tChecking the lateral (xy) coordinate data are sensible ... ', end='', flush=True)
            try:
                aspect_ratio = (xy_range[1] - xy_range[0]) / (xy_range[3] - xy_range[2])
#                data_ranges = max((xy_range[1] - xy_range[0]), (xy_range[3] - xy_range[2]), ps['ImageSize'])
               #  smallest_thing = min(xy_range[0], xy_range[1], xy_range[2], xy_range[3], 0)
            except ZeroDivisionError:
                aspect_ratio = 0
            
            if aspect_ratio < Sane_min_xy_ratio or aspect_ratio > Sane_max_xy_ratio:
                fn_etc.err_msg('x-range is ' + str(round(aspect_ratio,5)) + ' times the y-range!\n\t\tAcceptable limits for the x-range are ' + str(Sane_min_xy_ratio) + ' to ' + str(Sane_max_xy_ratio) + ' times the y-range.\n\t\tCheck z-column input or adjust the \'Sanity Check\' variables.')
                raise ValueError('Large difference between x and y data ... are the correct columns given?\nRemember to use zero indexing when specifying your columns!\nIf columns are correct then inspect the values used when performing the \'Sanity Check\'.')
                # if you really want to do analysis in very long thin RoIs then set DoSanityCheck to False in the original script.
#                
#            if data_ranges != ps['ImageSize']: # or smallest_thing < 0):
#                
#                fn_etc.err_msg('Points found outside image bounds')
#                raise ValueError('One or more x or y coordinates are located outside of the stated image boundary.')
            else:
                fn_etc.ok_msg('Input xy data looks good!')
            
            if ps['three_dee'] and np.isnan(datatable[:,ps['zCol']]).any():
                print('SanityCheck: \t Checking the axial (z) data are sensible ... ', end='', flush=True)

                try:
                    zx_aspect_ratio = (z_range[1] - z_range[0]) / (xy_range[1] - xy_range[0])
                except ZeroDivisionError:
                    zx_aspect_ratio = 0
                try:
                    zy_aspect_ratio = (z_range[1] - z_range[0]) / (xy_range[3] - xy_range[2])
                except ZeroDivisionError:
                    zy_aspect_ratio = 0
                # here we expect our z data to be much more restricted than our xy data
                # we can catch incorrect xyz column specifications at this point of processing.
                # if your data are going to routinely fail these checkes (e.g. you are not using SMLM data as input)
                # then you can disable the sanity checks at the top of this script.
                if zx_aspect_ratio > Sane_max_zx_ratio:
                    fn_etc.err_msg('z-range is ' + str(round(zx_aspect_ratio,5)) + ' times the x-range - check z-column input!\n\t\tAcceptable maximum for the z-range is ' + str(Sane_max_zx_ratio) + ' times the x-range.\n\t\tCheck z-column input or adjust the \'Sanity Check\' variables.')
                    raise ValueError('Large difference in scale between x and z data ... are the correct columns given?\nRemember to use zero indexing when specifying your columns!\nIf columns are correct then inspect the values used when performing the \'Sanity Check\'.')
                elif zy_aspect_ratio > Sane_max_zx_ratio:
                    fn_etc.err_msg('z-range is ' + str(round(zy_aspect_ratio,5)) + ' times the y-range - check z-column input!\n\t\tAcceptable maximum for the z-range is ' + str(Sane_max_zy_ratio) + ' times the y-range.\n\t\tCheck z-column input or adjust the \'Sanity Check\' variables.')
                    raise ValueError('Large difference in scale between y and z data ... are the correct columns given?\nRemember to use zero indexing when specifying your columns!\nIf columns are correct then inspect the values used when performing the \'Sanity Check\'.')
                else:
                    fn_etc.ok_msg('Input z data looks good!')
            
            if SanityCheck_FirstOnly:
                doSanityCheck = False
            #
            # End of Sanity Check
            #
    
    #        p = np.zeros_like(datatable[:, ps['xCol']], dtype=bool)
    #        q = np.zeros_like(datatable[:, ps['yCol']], dtype=bool)
    #        
    #        p[np.unique(datatable[:, ps['xCol']], return_index=True)[1]] = True
    #        q[np.unique(datatable[:, ps['yCol']], return_index=True)[1]] = True
    #        
    #        pandq = [a or b for a, b in zip(p, q)]
    #        datatable = datatable[pandq[:], :]

        #duplicate xy screening
        if doFilterDuplicates:
            if ps['three_dee']:
                data_coordsonly = np.concatenate((datatable[:, ps['xCol'], None], datatable[:, ps['yCol'], None], datatable[:, ps['zCol'], None]), axis=1)
            else:
                data_coordsonly = np.concatenate((datatable[:, ps['xCol'], None], datatable[:, ps['yCol'], None]), axis=1)
            
            _, uniq_idx = np.unique(data_coordsonly, axis=0, return_index=True) # get the unique rows
            
            if uniq_idx.shape[0] < datatable.shape[0]:
                uniq_idx = np.sort(uniq_idx)
                datatable = datatable[uniq_idx,:]
                
                oldTotalPoints = TotalPointsThisImage
                TotalPointsThisImage = datatable.shape[0]
                DuplicatePointsRemoved = oldTotalPoints - TotalPointsThisImage
                
                if not doAddUID:
                    doAddUID = True
                    
                fn_etc.info_msg('Checked for duplicate points and removed ' + str(DuplicatePointsRemoved) + ' identical points.')
            else:
                DuplicatePointsRemoved = 0
                print('Checked for duplicate points and none were found')
            del data_coordsonly
        
        # save a preview image
        if SaveImages:
            # save an image although this may change if we detect simulated data
            ps['SaveMyImages'] = True 
            
            if 'SaveImagesForRepeat' in ps and SimuCell_SaveImagesFor.lower() == 'partial':
                # first test for simulated dataset as 'SaveImagesForRepeat' key is
                # only in the ps dict for simualated data.
                # If 'partial' is set then might need to not-save image for this data!
                
                # Check the file name for clues
                SimuCell_test = output_prefix.split('_')[-1] # the last element should be the repeat number
                SimuCell_Ints=''
                SimuCell_Chars=''
                
                # Separate the chars from the integers.
                # Expecting 'cellID(n)' at the end of simulated data filenames.
                for s in SimuCell_test:
                    if s.isdigit():
                        SimuCell_Ints = SimuCell_Ints + s
                    else:
                        SimuCell_Chars = SimuCell_Chars + s

                if not all((int(SimuCell_Ints) == ps['SaveImagesForRepeat'], SimuCell_Chars == 'cellID()')):
                    ps['SaveMyImages'] = False # this file isn't flagged for image-saving
                
                del(SimuCell_test, SimuCell_Ints, SimuCell_Chars)

        else:
            # do not save images
            ps['SaveMyImages'] = False


        if ps['SaveMyImages']:
            print('Saving image of all points... ', end='', flush=True)
            for ExportImage in ExportImageFormat:
                print('[' + ExportImage + ']... ', end='', flush=True)
                preview_fname = os.path.join(s1_prep_outputpath, output_prefix + '_preview.' + ExportImage)
                fn_etc.make_image(datatable, preview_fname, ps, cmap2ch)
            print('Done.')

            
        if doAppendIsoGrid:
            print('Salting data with isometric background points')
            import FuncIsoGrid as fn_isogrid
            iso_grid = fn_isogrid.iso_spots(boundary=[ps['xMin'], ps['xMax'], ps['yMin'], ps['yMax']], spacing=50)
            iso_appendage = np.empty((iso_grid.shape[0], datatable.shape[1], ))
            iso_appendage[:] = np.nan
            iso_appendage[:, ps['xCol']] = iso_grid[:, 0]
            iso_appendage[:, ps['yCol']] = iso_grid[:, 1]
            datatable = np.concatenate((datatable, iso_appendage))
            TotalPointsThisImage = datatable.shape[0] # update this from before
            del(iso_grid, iso_appendage)
            
            # recheck for duplicates (TODO move this to a function)
            if doFilterDuplicates:
                data_xy = np.concatenate((datatable[:, ps['xCol'], None], datatable[:, ps['yCol'], None]), axis=1)
                _, uniq_idx = np.unique(data_xy, axis=0, return_index=True)
                if uniq_idx.shape[0] < datatable.shape[0]:
                    uniq_idx = np.sort(uniq_idx)
                    datatable = datatable[uniq_idx,:]

                    oldTotalPoints = TotalPointsThisImage
                    TotalPointsThisImage = datatable.shape[0]

                    if not doAddUID:
                        doAddUID = True
                    fn_etc.info_msg('Re-screened for duplicate xy points: removed ' + str(oldTotalPoints - TotalPointsThisImage) + ' identical points.')
                else:
                    print('Re-screened for duplicate xy points: none were found')
                del data_xy
            
        if doAddUID:
            ps['UIDCol'] = np.shape(datatable)[1]  # will be with zero ref
            datatable = np.insert(datatable,
                                  ps['UIDCol'],
                                  list(range(1, len(datatable)+1)),
                                  axis=1)
            ps['TableHeaders'] = ps['TableHeaders']  + ps['InputFileDelimiter']  +  'pointUID'
            # print('UIDCol:\tAdded Unique ID to column ' + str(ps['UIDCol']))
    
        if not 'LabelIDCol' in ps and prepType == 'training':
            ps['LabelIDCol'] = np.shape(datatable)[1]
            ClassLabels_tmp = datatable[:, ps['ClusMembershipIDCol']] > 0
            ClassLabels_tmp = ClassLabels_tmp.astype(int)
            datatable = np.insert(datatable,
                                  ps['LabelIDCol'],
                                  ClassLabels_tmp,
                                  axis=1)
            ps['TableHeaders'] = ps['TableHeaders'] + ps['InputFileDelimiter']  + 'Class Label'
            print('LabelIDCol: \t Added Class Labels to column ' + str(ps['LabelIDCol']))

        # Very rough estimate of processing time -- this will vary with every machine
        # and depend on how many CPU cores you employ.
        # once you have a few files done you can work out some rough params and put
        # them in at the top as ProcTimeEst_ABC for your own edification...
        GuessProcTime = ((ProcTimeEst_A * TotalPointsThisImage * TotalPointsThisImage) + (ProcTimeEst_B * TotalPointsThisImage) + ProcTimeEst_C) 
        if GuessProcTime > 30:
            FancyProcTime = datetime.timedelta(seconds=np.ceil(GuessProcTime))
            print('Total of ' + str(TotalPointsThisImage) + ' points is expected to take ' + str(FancyProcTime) + ' to calculate distances.') # + GuessFinishTime.strftime('%H:%M:%S on %d %B'))
        else:
            print('Total of ' + str(TotalPointsThisImage) + ' points.')
        free_space = fn_etc.get_free_space(s1_prep_outputpath)
        GuessMemoryRequirements = (( (TotalPointsThisImage * (ps['FurthestFriend'] - ps['ClosestFriend'] + 1) * (datatable.shape[1] + 1)) * len(ps['ImageSize']) * 64) + (datatable.size * 64)) / 8  # float64 bytes required
        print(fn_etc.convert_size(GuessMemoryRequirements) + ' is required to store this image\'s xy data and distance measurements (' + fn_etc.convert_size(free_space) + ' available space)')       
        if GuessMemoryRequirements > free_space:
            print('\x1b[1;35;43m' + '\tPROBLEM!\t' + '\x1b[1;37;45m' + '\tInsufficient storage space...\t' + '\x1b[0m', flush=True)
            # raise ValueError('This images requires ' + fn_etc.convert_size(GuessMemoryRequirements) + ' to store distance measurements and there is ' + fn_etc.convert_size(free_space) + ' remaining on the drive containing the output folder:\n' + s1_prep_outputpath)
            print('\x1b[1;33;40m' + 'Free up at least ' + fn_etc.convert_size(GuessMemoryRequirements - free_space) + ' to continue processing this image.\nBe aware that subsequent images in the queue may also require additional storage space.' + '\x1b[0m', flush=True)
            ask_prepType = fn_etc.askforinput(
                            message = 'Enter Y when you have confirmed there is more space...',
                            errormessage= 'Press Y and press enter to continue\nor type Ctrl C to cancel processing here.',
                            defaultval= '',
                            isvalid = lambda v : v in ['Y','y','Yes','yes'])
    
        # Set up a memory mapped file to hold a copy of datatable. You can switch this
        # out to a regular in-memory variable if you wish but things can bog down
        # when large datatables are loaded. If you work with MemMaps then you at least
        # have a good excuse to buy some fast PCIe or SSD storage, right?
        datatable_mmap_partname = output_prefix + '_Data.MemMap'
        ps['datatable_mmap_fname'] = datatable_mmap_partname
        datatable_mmap_fname = os.path.join(s1_prep_outputpath, datatable_mmap_partname)
        
        datatable_mmapped = np.memmap(
                                     datatable_mmap_fname, 
                                     dtype='float64', 
                                     shape=datatable.shape, 
                                     mode='w+')
        datatable_mmapped[:] = datatable
        
        del datatable    # free up memory by dropping the original datatable
        _ = gc.collect() # and forcing garbage collection
    
        # Pre-allocate another writeable shared memory map files to contain 
        # the ouput of the parallel distance-to-NNs computation
        dists_mmap_partname = output_prefix + '_D[NN' + str(ps['ClosestFriend']) + '-' + str(ps['FurthestFriend']) + ']_Dists.MemMap'
        ps['dists_mmap_fname'] = dists_mmap_partname
        dists_mmap_fname = os.path.join(s1_prep_outputpath, dists_mmap_partname)
        dists_mmapped = np.memmap(
                                 dists_mmap_fname, 
                                 dtype=datatable_mmapped.dtype, 
                                 shape=(TotalPointsThisImage, ps['FurthestFriend'] - ps['ClosestFriend'] + 1, datatable_mmapped.shape[1] + 1), 
                                 mode='w+')
        #
        # perform the distance measurements.
        #
        # Output is 3D array with:
        #    each point (dim 0)
        #    it's FurthestFriend NN UIDs (dim 1)
        #    and the measurements (distances, matching x,y,etc from original data) (dim 2)
        # distances_only = dists_mmapped[:,:,0]
    
        start_time = time.time()        
#        # If you want to do single-thread processing do this:
#        for i in range(TotalPointsThisImage):
#            fn_distcalc.dnns_v3(datatable_mmapped, datatable_mmapped, ps, dists_mmapped, i)
#            if np.mod(i, 1000) == 0:
#                elapsed_time = time.time() - start_time
#                print(str(i) + ' done - ' + str(round(elapsed_time,2)) + ' sec')
        # Otherwise do this line for parallel processing
        if ps['three_dee']:
            Parallel(n_jobs=total_cpus, verbose=3)(delayed(fn_distcalc_3D.dnns_v3)(datatable_mmapped, datatable_mmapped, ps, dists_mmapped, i) for i in range(TotalPointsThisImage))
        else:
            Parallel(n_jobs=total_cpus, verbose=3)(delayed(fn_distcalc.dnns_v3)(datatable_mmapped, datatable_mmapped, ps, dists_mmapped, i) for i in range(TotalPointsThisImage))
        elapsed_time = time.time() - start_time
        print('Time for ' + str(TotalPointsThisImage) + ' points was ' + str(round(elapsed_time,2)) + ' seconds (' + str(round(TotalPointsThisImage / elapsed_time,2)) + ' points per second.)')
    
        if prepType == 'training':
            
            # calc the target vector = number of nearest neighbours of the same type (zero if the type is zero)
            
            #slice output to get just the clusterIDs
            cluster_IDs_only = dists_mmapped[:,:,ps['ClusMembershipIDCol'] + 1]
            target_vectors = np.zeros((cluster_IDs_only.shape[0],1), dtype='int')
            for v in range(cluster_IDs_only.shape[0]):
                
                my_type = datatable_mmapped[v,ps['ClusMembershipIDCol']]
                nearest_type = cluster_IDs_only[v,0]
                
                # only assign positive vectors to points which are in clusters and have neighbours in clusters
                if my_type != 0 and nearest_type != 0:
                    for x in range(0, ps['FurthestFriend']):
                        if cluster_IDs_only[v,x] != nearest_type:
                            break
                    target_vectors[v,0] = x
            
            # convert vectors to binary targets (0 = not clustered, 1 = clustered)
            # This will exclude some actually clustered points if they are closer to a NC point than to any other clustered point
            target_binary = np.array(target_vectors > 0, dtype='int')
            
            # get our target labels directly, if they exist
            target_labels = datatable_mmapped[:, ps['LabelIDCol'], None]
            
            # alternative binary targets from the original cluster IDs
        #    target_binary2 = np.array(datatable_mmapped[:, ps['ClusMembershipIDCol']] > 0, dtype='int').reshape(TotalPointsThisImage,1)
           
            ClusteredPointsThisImage = np.sum(target_binary)
            total_clusters_this_image = int(np.max(cluster_IDs_only))
            print('Image has ' + str(ClusteredPointsThisImage) + ' clustered points (from ' + str(total_clusters_this_image) + ' clusters) and ' + str(TotalPointsThisImage - ClusteredPointsThisImage) + ' non-clustered points.')
            
            class_types = np.unique(target_labels).shape[0]
            
            for c in range(class_types):
                total_this_type = sum(target_labels == c)
                print('\t' + str(total_this_type) + ' points of with label \'' + str(c) + '\'.')
            
            # save these processing stats to a log file; will be used to split the tables for training
            with open(os.path.join(s1_prep_outputpath, 'PrepLog.txt'), 'a+', newline='') as prep_log:
                writer = csv.writer(prep_log, delimiter='\t')
                writer.writerow([str(fileIdx + 1), str(TotalPointsThisImage), str(round(elapsed_time,2)), str(ClusteredPointsThisImage), str(TotalPointsThisImage - ClusteredPointsThisImage), str(total_clusters_this_image), str(DuplicatePointsRemoved)])
            
    #        pickle.dump((dists_mmapped, 
    #                     ps, 
    #                     target_vectors, 
    #                     target_binary), 
    #                     open(os.path.join(s1_prep_outputpath, output_prefix + '_dists[NN' + str(ps['ClosestFriend']) + '-' + str(ps['FurthestFriend']) + ']-Done_pparNEW.pkl'), 'wb'), 
    #                     protocol=4)
    
            # dump target vectors 
            target_vectors_mmap_partname = output_prefix + '_TargetVectors.MemMap'
            ps['target_vectors_mmap_fname'] = target_vectors_mmap_partname
            ps['VectorsDumpShape'] = target_vectors.shape
            target_vectors_mmap_fname = os.path.join(s1_prep_outputpath, target_vectors_mmap_partname)
            target_vectors_mmapped = np.memmap(
                                             target_vectors_mmap_fname, 
                                             dtype='int', 
                                             shape=target_vectors.shape, 
                                             mode='w+')
            target_vectors_mmapped[:] = target_vectors
            
            # dump target binary
            target_binary_mmap_partname = output_prefix + '_TargetBinary.MemMap'
            ps['target_binary_mmap_fname'] = target_binary_mmap_partname
            ps['BinaryDumpShape'] = target_binary.shape
            target_binary_mmap_fname = os.path.join(s1_prep_outputpath, target_binary_mmap_partname)
            target_binary_mmapped = np.memmap(
                                             target_binary_mmap_fname, 
                                             dtype='int', 
                                             shape=target_binary.shape, 
                                             mode='w+')
            target_binary_mmapped[:] = target_binary
            
            # dump target labels
            target_labels_mmap_partname = output_prefix + '_TargetLabels.MemMap'
            ps['target_labels_mmap_fname'] = target_labels_mmap_partname
            ps['LabelsDumpShape'] = target_labels.shape
            target_labels_mmap_fname = os.path.join(s1_prep_outputpath, target_labels_mmap_partname)
            target_labels_mmapped = np.memmap(
                                             target_labels_mmap_fname, 
                                             dtype='int', 
                                             shape=target_labels.shape, 
                                             mode='w+')
            target_labels_mmapped[:] = target_labels
            
        elif prepType == 'novel':
            
            with open(os.path.join(s1_prep_outputpath, 'PrepLog.txt'), 'a+', newline='') as prep_log:
                writer = csv.writer(prep_log, delimiter='\t')
                writer.writerow([str(fileIdx + 1), str(TotalPointsThisImage), str(round(elapsed_time,2)), str(DuplicatePointsRemoved)])
                
    #        pickle.dump((dists_mmapped, 
    #             ps), 
    #             open(os.path.join(s1_prep_outputpath, output_prefix + '_dists[NN' + str(ps['ClosestFriend']) + '-' + str(ps['FurthestFriend']) + ']-Done_pparNEW.pkl'), 'wb'), 
    #             protocol=4)
           
        # Update and Dump ProcSettings as JSON
        ps['DistsDumpShape'] = dists_mmapped.shape
        ps['DataDumpShape'] = datatable_mmapped.shape
        ps['FilePrefix'] = output_prefix
        
        json_fname = os.path.join(s1_prep_outputpath, output_prefix + '_dists[NN' + str(ps['ClosestFriend']) + '-' + str(ps['FurthestFriend']) + ']-ProcSettings.json')
        with open(json_fname, 'w') as file:
            file.write(json.dumps(ps, indent=4))
    
#        print('Finished image ' + str(fileIdx + 1) + ' at ' + datetime.datetime.now().strftime('%H:%M:%S on %d %B'))
        
        # cleanup temp folders
        del dists_mmapped, datatable_mmapped
        _ = gc.collect()
        
        print('Finished file ' + str(fileIdx + 1) + ' - ' + ps['FilePrefix'] + ' at ' + datetime.datetime.now().strftime('%H:%M:%S on %d %B') + '\n')
    ###
    #####
        # End of per-fileIdx processing
    #####
    ###

    fn_etc.ok_msg('Finished data preparation for all images.')
    print('The input folder was\t' + inputpath)
    print('The output folder was\t' + s1_prep_outputpath)
    if EmailUponCompletion:
        print(fn_etc.send_jobdone_email('Data Preparation on folder ' + s1_prep_outputpath, ' - Data Prep done'))
    print('-------------------------------------------------\n' + \
                      '\t\t\tCompleted!' + \
          '\n-------------------------------------------------')
