#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
If you don't want to draw freehand ROIs then this will draw out square ROIs for each file.

@author: dave
"""

import os
import numpy as np
from natsort import natsorted
import json

proc_wd = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print('Changed working directory to ' + proc_wd)

import FuncEtc as fn_etc

roi_sizes = list((3000, 6000)) # width of each square ROI

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
    
    default_outfolder = os.path.abspath(os.path.join(inputpath, 'square_rois'))
    s1_prep_outputpath = fn_etc.askforinput(
            message = 'Enter the name of the output folder',
            errormessage= 'The output folder must be named',
            defaultval= default_outfolder,
            isvalid = lambda v : len(v) > 0)
   
    # get a list of input files from the given inputfolder
    files = natsorted([i for i in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, i)) and ps['InputFileExt'] in i])
    total_files = np.shape(files)[0]
    
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
    for fileIdx in range(0, total_files):

        current_file = files[fileIdx]
        output_prefix = os.path.splitext(current_file)[0]
        fn_etc.info_msg(str(fileIdx + 1) + ' of ' + str(total_files) + '\t' + current_file)

        for roi_id, roi_size in enumerate(roi_sizes):
            
            print('Saving ROI ' + str(roi_id + 1) + ' - ' + str(roi_size) + ' nm square')
            centre_x = ps['xMax']  / 2
            centre_y = ps['yMax']  / 2
            half_roi_width = roi_size / 2
            
            roi_min_x = centre_x - half_roi_width
            roi_max_x = centre_x + half_roi_width
            
            roi_min_y = centre_y - half_roi_width
            roi_max_y = centre_y + half_roi_width
            
            roi_out = [[roi_min_x, roi_min_y],
                       [roi_min_x, roi_max_y],
                       [roi_max_x, roi_max_y],
                       [roi_max_x, roi_min_y],
                       [roi_min_x, roi_min_y]]
            
            json_fname = os.path.join(s1_prep_outputpath, output_prefix + '_roi_' + str(roi_id + 1) + '.json')
            with open(json_fname, 'w') as file:
                file.write(json.dumps(roi_out, indent=4))