#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduces the number of points in SMLM images to a given maximum count or fraction.

This script takes a folder of SMLM files and, for each file, saves a copy of it up 
to the specified maximum number of points in the output folder.

The number of points is given either as an exact value or a percentage of the 
initial total points.

Images with less than the specified maximum are simply copied to the output folder.

@author: Dave

"""

import os
import numpy as np
from natsort import natsorted
import gc
import datetime
import json

proc_wd = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print('Changed working directory to ' + proc_wd)

import FuncEtc as fn_etc

# Screen for identical xy coordinate pairs
# If found only the first pair will be retained and the other removed.
# Identical points give rise to zero-length neighbour distances which can break 
# things further down the line.
doFilterDuplicates = True 

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
#  Files = [0,1,2,3,4,5,6,7,8,9]
#  starting_index = 0  and finishing_index = 5 >>> Processes Files 0,1,2,3,4
#  starting_index = 8  and finishing_index = 3 >>> Processes Files 8,7,6,5,4

if __name__ == '__main__':
    
    # Initial processing settings (ProcSettings) are loaded from a JSON file
    good_json = False
    default_json_file = ''
    
    while not good_json:
        
        if 'input_PrepJSON' in locals():
            default_json_file = input_PrepJSON # recycle the previous input

        input_PrepJSON = fn_etc.askforinput(
            message = 'Full path to JSON file describing the data',
            errormessage = 'The file you provided does not exist or you supplied a path only. Check that your path includes the file you want and try again.',
            defaultval = default_json_file,
            isvalid = lambda v : os.path.isfile(v))
        
        with open(input_PrepJSON, 'r') as file:
            ps = json.loads(file.read())
        
        fn_etc.info_msg('Imported JSON variables:')
        print(' │')
        print(' ├─InputFileDelimiter:\t' + ps['InputFileDelimiter'])
        print(' ├─InputFileExt:\t' + ps['InputFileExt'])
        print(' │')
        print(' ├─xCol:\t\t' + str(ps['xCol']))
        print(' ├─yCol:\t\t' + str(ps['yCol']))
        print(' ├─ClusMembershipIDCol:\t' + str(ps['ClusMembershipIDCol']))
        print(' ├─ChanIDCol:\t\t' + str(ps['ChanIDCol']))
        print(' ├─UIDCol:\t\t' + str(ps['UIDCol']))
        print(' │')
        print(' ├─AutoAxes:\t\t' + str(ps['AutoAxes']))
        if ps['AutoAxes']:
            print(' ├─AutoAxesNearest:\t' + str(ps['AutoAxesNearest']))
            print(' ├─ImageSize:\t\tTo be determined')
            print(' ├─xMin:\t\tTo be determined')
            print(' ├─xMax:\t\tTo be determined')
            print(' ├─yMin:\t\tTo be determined')
            print(' ├─yMax:\t\tTo be determined')
        else:
            print(' ├─AutoAxesNearest:\tNot applicable')
            print(' ├─ImageSize:\t\t' + str(ps['ImageSize']))
            print(' ├─xMin:\t\t' + str(ps['xMin']))
            print(' ├─xMax:\t\t' + str(ps['xMax']))
            print(' ├─yMin:\t\t' + str(ps['yMin']))
            print(' ├─yMax:\t\t' + str(ps['yMax']))
        print(' │')
        print(' ├─ClosestFriend:\t' + str(ps['ClosestFriend']))
        print(' └─FurthestFriend:\t' + str(ps['FurthestFriend']))

        verify_good_json = fn_etc.askforinput(
                message = 'Are these settings correct? Enter \'Y\' to proceed or enter \'N\' to select another JSON file (or the same file, after you have edited it with the correct settings)',
                errormessage= 'Type Y or N',
                defaultval= 'y',
                isvalid = lambda v : v.lower() in ['y','n','yes','no'])
        
        if verify_good_json.lower() in ['y','yes']:
            print('JSON file accepted.')
            good_json = True
        else:
            print('JSON file rejected.')
            
    
    default_input_path = os.path.dirname(input_PrepJSON)
       
    # get the data from the folder
    inputpath = fn_etc.askforinput(
            message = 'Enter the path of the folder containing ' + ps['InputFileExt'] + ' data tables',
            errormessage= 'The folder you provided does not exist or you have provided the path to a file.',
            defaultval= default_input_path,
            isvalid = lambda v : os.path.isdir(v))

    reduction_method = fn_etc.askforinput(
        message = 'Do you want to reduce data to [1]-Maximum total points or [2]-Fraction of original points? (Enter 1 or 2)',
        errormessage= 'Type the number 1 or 2 and press enter',
        defaultval= '2',
        isvalid = lambda v : v in ['1','2'])
    
    if reduction_method in ['1']:
        max_points_per_set = fn_etc.askforinput(
                message = 'Enter the maximum number of points to be retained from each dataset',
                errormessage= 'Please enter a non-zero positive integer',
                defaultval= '100000',
                isvalid = lambda v: v.isdigit() and int(v) >= 1)
        max_points_per_set = int(max_points_per_set)
        fraction_points_per_set = 0
        default_outfolder = os.path.join(inputpath, 'Reduced_' + str(max_points_per_set) + '_Pts_Max')
    elif reduction_method in ['2']:
        fraction_points_per_set = fn_etc.askforinput(
                message = 'Enter the fraction of points to be retained from each dataset',
                errormessage= 'Please enter a number which is greater than zero and less than one',
                defaultval= '0.5',
                isvalid = lambda v: v.replace('.','').isdigit() and float(v) > 0 and float(v) < 1)
        fraction_points_per_set = float(fraction_points_per_set)
        max_points_per_set = 0
        default_outfolder = os.path.join(inputpath, 'Reduced_' + str(fraction_points_per_set) + 'x_Pts')
    

    s1_prep_outputpath = fn_etc.askforinput(
            message = 'Enter the name of the output folder',
            errormessage= 'The output folder must be named',
            defaultval= os.path.abspath(default_outfolder),
            isvalid = lambda v : len(v) > 0)

    # get a list of input files from the given inputfolder
    files = natsorted([i for i in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, i)) and ps['InputFileExt'] in i])
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

    proceed_with_processing = fn_etc.askforinput(
        message = 'When you are ready to proceed type P and Enter (or X to cancel everything and exit)',
        errormessage= 'Type P or X',
        defaultval= 'P',
        isvalid = lambda v : v.lower() in ['p','x'])
    
    if proceed_with_processing.lower() in ['p']:
        print('Rightyo, off we go...')
    else:
        print("That's ok. Maybe next time?")
        exit()

    #make the folder for the output data
    if not os.path.exists(s1_prep_outputpath):
        os.makedirs(s1_prep_outputpath)
        
    # process all the files
    for fileIdx in range(starting_index, finishing_index):

        current_file = files[fileIdx]
        output_prefix = os.path.splitext(current_file)[0]
        fn_etc.info_msg(str(fileIdx + 1) + ' of ' + str(total_files) + '\t' + current_file)
        print('Loading data...', end='', flush=True)
        datatable = np.genfromtxt(os.path.join(inputpath, current_file),
                                  delimiter=ps['InputFileDelimiter'],
                                  skip_header=1) # names=True

        # will be exporting as tab-delimited from here, so swap out original delimiters in the header for tabs
        with open(os.path.join(inputpath, current_file), 'r') as f:
            ps['TableHeaders'] = f.readline().strip()
    
        TotalPointsThisImage = datatable.shape[0]
        print('Done (' + str(TotalPointsThisImage) + ' points)')
        
        #duplicate xy screening
        if doFilterDuplicates:
            data_xy = np.concatenate((datatable[:, ps['xCol'], None], datatable[:, ps['yCol'], None]), axis=1)
            _, uniq_idx = np.unique(data_xy, axis=0, return_index=True)
            if uniq_idx.shape[0] < datatable.shape[0]:
                uniq_idx = np.sort(uniq_idx)
                datatable = datatable[uniq_idx,:]

                oldTotalPoints = TotalPointsThisImage
                TotalPointsThisImage = datatable.shape[0]
                DuplicatePointsRemoved = oldTotalPoints - TotalPointsThisImage

            else:
                DuplicatePointsRemoved = 0
                print('Screened for duplicate xy points: none were found')
            del data_xy

        
        if fraction_points_per_set > 0:
            max_points_per_set = int(fraction_points_per_set * TotalPointsThisImage)
       
        if TotalPointsThisImage <= max_points_per_set:
            print('This file has the same or fewer points than the specified maximum (' + str(max_points_per_set) + ') points. It will be copied to the output folder as-is.')
            datatable_reduced = datatable
            reduced_output_fname = os.path.join(s1_prep_outputpath, output_prefix + '_copied' + ps['InputFileExt'])
        else:
            print('Choosing ' + str(max_points_per_set) + ' points for the reduced dataset (' + str(round((max_points_per_set / TotalPointsThisImage) * 100, 2)) + '% of original total points)')
            # keep a random subset of the points up to the maximum number of points specified
            keepers_idx = np.random.choice(datatable.shape[0], max_points_per_set, replace=False)
            datatable_reduced = datatable[keepers_idx]
            reduced_output_fname = os.path.join(s1_prep_outputpath, output_prefix + '_reduced' + ps['InputFileExt'])
        
        # save the reduced datatable to the output folder
        print('Saving reduced dataset...', end='', flush=True)
        np.savetxt(reduced_output_fname, datatable_reduced, delimiter=ps['InputFileDelimiter'], header=ps['TableHeaders'], fmt="%s")
        print('Done')
        
        # clean up
        _ = gc.collect()
        
        print('Finished file ' + str(fileIdx + 1) + ' - ' + output_prefix + ' - at ' + datetime.datetime.now().strftime('%H:%M:%S on %d %B') + '\n')
    ###
    #####
        # End of per-fileIdx processing
    #####
    ###

    fn_etc.ok_msg('Finished data preparation for all images.')
    print('The input folder was\t' + inputpath)
    print('The output folder was\t' + s1_prep_outputpath)
    print('-------------------------------------------------\n\tCompleted!\n-------------------------------------------------')
