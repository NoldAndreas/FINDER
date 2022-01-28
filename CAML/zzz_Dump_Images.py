#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make scatterplot images of SMLM data tables in the supplied folder.

@author: dave
"""

import os
import numpy as np
from natsort import natsorted

proc_wd = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print('Changed working directory to ' + proc_wd)

import FuncEtc as fn_etc


# ========================================================================
# Variables
# ========================================================================

ps=dict()

ps['InputFileExt'] = '.tsv'
ps['InputFileDelimiter'] = '\t'
ps['xCol'] = 0
ps['yCol'] = 1

ps['scores_col'] = 0
ps['labels_col'] = 2
ps['clustID_col'] = 2

ps['doSetUpAxes'] = False
ps['AutoAxesNearest'] = 1000
ps['xMin'] = 0
ps['xMax'] = 40000
ps['yMin'] = 0
ps['yMax'] = 40000
ps['ImageSize'] = 40000

doSanityCheck = True
SanityCheck_FirstOnly = True

PreviewImageFormat = 'png' # file extension of the saved images. Use 'png', 'jpg', 'svg', 'eps'.

ColourScheme = 'light' # or 'light'

# Dark scheme, good for presentations
dark_plotfgcolor = (1,1,1)                  # text & line colour for all plots
dark_plotbgcolor = (0.1765,0.1765,0.1765)    # background colour for all plots

# Light scheme, good for printing
light_plotfgcolor = (0,0,0)                  # text & line colour for all plots
light_plotbgcolor = (1,1,1)                   # background colour for all plots

starting_index = 0   # normally zero
finishing_index = 0  # normally zero

np.random.seed(270879)                  # fix random seed for reproducibility

cmapfile = './cmaps/candybright_256_rgb.txt' # linear colourmap, e.g. for model score map
cmap2chfile = './cmaps/two_channel_bluyel.txt' # two-phase colormap, e.g. for binary classification map

# ========================================================================
# Begin main
# ========================================================================
if __name__ == '__main__':

    cmap = fn_etc.load_colormap(cmapfile, flip=False)
    cmap2ch = fn_etc.load_colormap(cmap2chfile, flip=False)

    if ColourScheme == 'light':
        plotfgcolor = light_plotfgcolor
        plotbgcolor = light_plotbgcolor
    else:
        plotfgcolor = dark_plotfgcolor
        plotbgcolor = dark_plotbgcolor        

    default_input_path = ''
        
    input_dir = fn_etc.askforinput(
            message = 'Folder with prepared files (.MemMap and .json)',
            errormessage= 'The folder you provided does not exist or you have supplied a file instead of a folder.',
            defaultval= default_input_path,
            isvalid = lambda v : os.path.isdir(v))

    # specify an output folder ... default is to step up one level from the input so we keep things together instead of nested
    output_dir = fn_etc.askforinput(
            message = 'Output folder',
            errormessage= 'The dataset must be named for an output folder to be created!',
            defaultval= os.path.abspath(os.path.join(input_dir, '..', '0_images')),
            isvalid = lambda v : len(v) > 0)
 
    ## get a list of the files to process from the given folder
    input_files = natsorted([i for i in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, i)) and ps['InputFileExt'] in i])

    if np.shape(input_files)[0] > 0:
        total_files = np.shape(input_files)[0]
    else:
        raise ValueError('No files to process! Check you have a) given the correct folder and b) this folder contains files prepared by the \'Data preparation\' processing step.')

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
            finishing_index = total_files
            print('Finishing Index has been reset and all files in the folder will be processed.')
        else:
            print('Keeping the current index. Processing will end once File ' + str(finishing_index) + ' is done.')
    else:
        finishing_index = total_files

    # wait here until we are given the go-ahead
    proceed_with_processing = fn_etc.askforinput(
        message = 'When you are ready to proceed type P and Enter (or X to cancel everything and exit)',
        errormessage= 'Type P or X',
        defaultval= 'P',
        isvalid = lambda v : v.lower() in ['p','x'])
    
    if proceed_with_processing.lower() in ['p']:
        print('Rightyo, off we go...\n')
    else:
        print("That's ok. Maybe next time?")
        exit()

    #make the folder for the output data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ========================================================================
    #    For each file ... 
    # ========================================================================
    #        
    # using starting_index we can easily pick up again in case of stoppages
    for fileIdx in range(starting_index, finishing_index):
        
        ps_current = dict(ps)
        current_file = input_files[fileIdx]
        output_prefix = os.path.splitext(current_file)[0]
        fn_etc.info_msg(str(fileIdx + 1) + ' of ' + str(total_files) + '\t' + current_file)
        print('Loading data...', end='', flush=True)
        datatable = np.genfromtxt(os.path.join(input_dir, current_file),
                                  delimiter=ps_current['InputFileDelimiter'],
                                  skip_header=1) # names=True
        # usecols=(xCol,yCol) is an option here but ultimately we want to regenerate the full data table with the addition of clustering info columns

        TotalPointsThisImage = datatable.shape[0]
        print('Done (' + str(TotalPointsThisImage) + ' points)')
        

        
        ps_current['FilePrefix'] = os.path.splitext(current_file)[0]
        
        # auto image boundaries; avoid plotting cropped regions on a full-sized field.
        if ps_current['doSetUpAxes'] == True:
            
            # get xy range of the data
            xmin = np.min(datatable[:,ps_current['xCol']])
            xmax = np.max(datatable[:,ps_current['xCol']])
            ymin = np.min(datatable[:,ps_current['yCol']])
            ymax = np.max(datatable[:,ps_current['yCol']])
            
            ps_current['xMin'] = np.floor(xmin / ps_current['AutoAxesNearest'] ) * ps_current['AutoAxesNearest']
            ps_current['xMax'] = np.ceil(xmax / ps_current['AutoAxesNearest'] ) * ps_current['AutoAxesNearest']
            ps_current['yMin'] = np.floor(ymin / ps_current['AutoAxesNearest'] ) * ps_current['AutoAxesNearest']
            ps_current['yMax'] = np.ceil(ymax / ps_current['AutoAxesNearest'] ) * ps_current['AutoAxesNearest']
            
            ps_current['ImageSize'] = np.max((ps_current['xMax'] - ps_current['xMin'], ps_current['yMax'] - ps_current['yMin']))
            ps_current['AutoAxes'] = False
            print('AutoAxes: \t Set image boundaries to [ ' + str(ps_current['xMin']) + ', ' + str(ps_current['xMax']) + ', ' + str(ps_current['yMin']) + ', ' + str(ps_current['yMax']) + ' ]')

        # Start of Sanity Check
        if doSanityCheck:
            print('SanityCheck: \t X data in column ' + str(ps_current['xCol']) )
            print('SanityCheck: \t Y data in column ' + str(ps_current['yCol']) )
            xy_range = [np.floor(min(datatable[:, ps_current['xCol']])),
                        np.ceil(max(datatable[:, ps_current['xCol']])), 
                        np.floor(min(datatable[:, ps_current['yCol']])), 
                        np.ceil(max(datatable[:, ps_current['yCol']]))
                        ]

            print('SanityCheck: \t Checking the data makes sense ... ', end='', flush=True)
            try:
                aspect_ratio = (xy_range[1] - xy_range[0]) / (xy_range[3] - xy_range[2])
            except ZeroDivisionError:
                aspect_ratio = 0
            
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                fn_etc.err_msg('Extreme xy-range aspect ratio (' + str(round(aspect_ratio,5)) + ') - check xy-column input...')
                raise ValueError('Large difference between x and y data ... are the correct columns given?\nRemember to use zero indexing when specifying your columns!')
                # if you really want to do analysis in very long thin RoIs then set DoSanityCheck to False in the original script.
            else:
                fn_etc.ok_msg('Input data looks good!')
            
            # Check for invalid input data in xy cols.
            if np.isnan(datatable[:,ps_current['xCol']]).any():
                fn_etc.err_msg('NaNs found in your input data x-coord column.')
                raise ValueError('Coordinate xy data must be real numbers, not NaNs.')
            elif np.isnan(datatable[:,ps_current['yCol']]).any():
                fn_etc.err_msg('NaNs found in your input data y-coord column.')
                raise ValueError('Coordinate xy data must be real numbers, not NaNs.')
            
            if SanityCheck_FirstOnly:
                doSanityCheck = False
            # End of Sanity Check


        # convert clusterIDs into point labels, if they are coming from the same column.
        if ps_current['labels_col'] == ps_current['clustID_col']:
            ClusteredLabels = datatable[:,ps_current['clustID_col']] > 0
            datatable = np.concatenate((datatable, ClusteredLabels[:, None]), axis = 1)
            ps_current['labels_col'] = np.shape(datatable)[1] - 1

        # save images
       
        # a flat image of the points without any colouring
        ps_temp = dict(ps_current)
        ps_temp['BackGroundColor'] = plotbgcolor
        ps_temp['ForeGroundColor'] = plotfgcolor
        if ps_temp['ImageSize'] / 50 < 300:
            ps_temp['OutputScaling'] = 1
            ps_temp['PointSize'] = 10
        else:
            ps_temp['OutputScaling'] = 100
            ps_temp['PointSize'] = 1.0

        print('Saving image of flat points... ', end='', flush=True)
        fn_etc.make_image(datatable, os.path.join(output_dir, ps_current['FilePrefix'] + '_PointsOnly.' + PreviewImageFormat), ps_temp, None)
        print('Done.')

        # points coloured by their ML scores
        if ps_current['scores_col'] > 0:
            ps_temp['ClusMembershipIDCol'] = ps_current['scores_col']
            ps_temp['PointsMinValue'] = 0
            ps_temp['PointsMaxValue'] = 1
            print('Saving image of points coloured by model\'s score... ', end='', flush=True)
            fn_etc.make_image(datatable, os.path.join(output_dir, ps_current['FilePrefix'] + '_ByScore.' + PreviewImageFormat), ps_temp, cmap)
            print('Done.')
        
        # points coloured by the label (true or assigned)
        if ps_current['labels_col'] > 0:
            ps_temp['ClusMembershipIDCol'] = ps_current['labels_col']
            ps_temp['PointsMinValue'] = 0
            ps_temp['PointsMaxValue'] = 1
            print('Saving image of points colours by classification label... ',end='', flush=True)
            fn_etc.make_image(datatable, os.path.join(output_dir, ps_current['FilePrefix'] + '_ByLabel.' + PreviewImageFormat), ps_temp, cmap2ch)
            print('Done.')

        # points coloured by their cluster membership (true or assigned)
        if ps_current['clustID_col'] > 0:
            ps_temp['ClusMembershipIDCol'] = ps_current['clustID_col']
            del ps_temp['PointsMinValue']
            del ps_temp['PointsMaxValue']
            print('Saving image of points colours by cluster membership... ',end='', flush=True)
            fn_etc.make_image(datatable, os.path.join(output_dir, ps_current['FilePrefix'] + '_ByClusterID.' + PreviewImageFormat), ps_temp, cmap)
            print('Done.')

        print('Finished file ' + str(fileIdx + 1) + ' - ' + ps_current['FilePrefix'] + '\n')
    # End of per-fileIdx processing

    print('-------------------------------------------------\n' + \
                      '\t\t\tCompleted!' + \
          '\n-------------------------------------------------')
    print('The input folder was\t' + input_dir)
    print('The output folder was\t' + output_dir)