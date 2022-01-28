#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates confusion matrices on evaluated data where the 'true' answer is also 
known (i.e. simulated or annotated datasets)

Also generates a summary csv for assembly accuracy heatmap plots.

@author: dave

"""

import os
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

proc_wd = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print('Changed working directory to ' + proc_wd)

import FuncEtc as fn_etc

# ========================================================================
#  User-editable variables
# ========================================================================

cmapfile = 'viridis'                      # Continuous colormap
plotbgcolor = (0.1765, 0.1765, 0.1765)    # Background color for scatter plots
plottxtcolor = (1, 1, 1)                  # Text color for scatter plots

# Input file parameters - these are with reference to the annotated data tables
# saved after model evaluation.
TrueCluster_Col = 2             # Ground-truth values in this column of the data
ReportedCluster_Col = 6         # Model-reported values are in this column
InputFileExt = '.tsv'           # file extension of input datatables
InputFileDelimiter = '\t'       # data-delimiter of input datatables


# ========================================================================
#  End of user-editable variables
# ========================================================================

if __name__ == '__main__':

    cmap = fn_etc.load_colormap(cmapfile)
    
    # get the data from the folder
    inputpath = fn_etc.askforinput(
            message = 'Enter the path of the folder containing model-classified ' + InputFileExt + ' data tables',
            errormessage= 'The folder you provided does not exist or you have provided the path to a file.',
            defaultval= '',
            isvalid = lambda v : os.path.isdir(v))
    
    ##### Column for True Values #####
    default_TrueCluster_Col = str(TrueCluster_Col)
    TrueCluster_Col = fn_etc.askforinput(
        message = 'Column containing ground-truth cluster labels (\'ClusterUID\')',
        errormessage= 'An integer greater than or equal to zero is required.',
        defaultval= default_TrueCluster_Col,
        isvalid = lambda v: v.isdigit() and int(v) >= 0)
    TrueCluster_Col = int(TrueCluster_Col)
    
    ##### Column for Found Values #####
    default_ReportedCluster_Col = str(ReportedCluster_Col)
    ReportedCluster_Col = fn_etc.askforinput(
        message = 'Column containing model-reported cluster labels (\'labels\')',
        errormessage= 'An integer greater than or equal to zero is required.',
        defaultval= default_ReportedCluster_Col,
        isvalid = lambda v: v.isdigit() and int(v) >= 0)
    ReportedCluster_Col = int(ReportedCluster_Col)
    
    # get a list of input files from the given inputfolder
    files = natsorted([i for i in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, i)) and InputFileExt in i])
    total_files = np.shape(files)[0]
    
    current_index = 0 # set to restart a run at the specified value or 0 for normal operation
    
    confusion_box = np.zeros((total_files,9))
    confusion_headers = 'fileIdx,Total Points,Total Correct 0,Total Incorrect 0,Total Correct 1,Total Incorrect 1,Percent Correct 0,Percent Correct 1,Percent Correcy Any'
    
    simucell_tracker = np.zeros((total_files,7))
    simucell_headers = 'PpMS,PpC,PC,r_min,r_max,CpMS,cellID'
    
    confusion_pool = np.zeros((1,2)) # ground-truth classification, model-classification ... init with a dummy row

    
    # process all the files
    for fileIdx in range(current_index, total_files):
    
        current_index = fileIdx
        current_file = files[fileIdx]
        output_prefix = os.path.splitext(current_file)[0]
        fn_etc.info_msg(str(current_index + 1) + ' of ' + str(total_files) + '\t' + current_file)
        
        splitter = fn_etc.multisplit((',', '_', '(', ')', '-'),current_file)

        if splitter[1] == 'PpMS' and splitter[12] == 'cellID':
            # we have a simucell so we can take apart its name
            simucell_tracker[fileIdx,:] = (splitter[2], splitter[4], splitter[6], 
                                           splitter[8], splitter[9], splitter[11], 
                                           splitter[13])
            
        # load only the ground-truth and model-reported clustering data
        datatable = np.genfromtxt(os.path.join(inputpath, current_file),
                                  delimiter=InputFileDelimiter,
                                  usecols=(TrueCluster_Col, ReportedCluster_Col), 
                                  skip_header=1)
        
        # convert cluster UID to label
        datatable[:,0] = datatable[:,0] > 0
        
        # add it to the pool
        confusion_pool = np.concatenate((confusion_pool, datatable))
        
        # get confusion matrix
        confusionmatrix_data = confusion_matrix(datatable[:,0], datatable[:,1], labels=[0,1])
        
        total_true_label_0 = np.sum(datatable[:,0] == 0)
        total_true_label_1 = np.sum(datatable[:,0] == 1)
        
        confusion_box[fileIdx, 0] = fileIdx + 1
        confusion_box[fileIdx, 1] = datatable.shape[0]
        confusion_box[fileIdx, 2] = confusionmatrix_data[0,0]
        confusion_box[fileIdx, 3] = confusionmatrix_data[1,0]
        confusion_box[fileIdx, 4] = confusionmatrix_data[1,1]
        confusion_box[fileIdx, 5] = confusionmatrix_data[0,1]
        
        if total_true_label_0 > 0:
            
            confusion_box[fileIdx, 6] = confusionmatrix_data[0,0] / total_true_label_0
        else:
            confusion_box[fileIdx, 6] = np.nan
            
        if total_true_label_1 > 0:
            confusion_box[fileIdx, 7] = confusionmatrix_data[1,1] / total_true_label_1
        else:
            confusion_box[fileIdx, 7] = np.nan
    
        confusion_box[fileIdx, 8] = (confusionmatrix_data[0,0] + confusionmatrix_data[1,1]) / datatable.shape[0]
    
    # per file confusion matrix
    #    model_summary_fname = ''
    #    
    #    # Plot non-normalized confusion matrix
    #    fig_AbsCM = fn_etc.plot_confusion_matrix(confusionmatrix_data, classes=['NC', 'C'],
    #                          cmap=cmap, plottxtcolor=plottxtcolor, plotbgcolor=plotbgcolor, normalize=False,
    #                          title='Absolute Confusion matrix')
    #    
    #    fig_AbsCM.savefig(model_summary_fname + ' - Abs Confusion Matrices.png',
    #                dpi=300,
    #                bbox_inches=0,
    #                facecolor=plotbgcolor,
    #                edgecolor='none',
    #                transparent=True)
    #    plt.close()
    #    
    #    # Plot normalized confusion matrix
    #    fig_NormCM =fn_etc.plot_confusion_matrix(confusionmatrix_data, classes=['NC', 'C'],
    #                          cmap=cmap, plottxtcolor=plottxtcolor, plotbgcolor=plotbgcolor, normalize=True,
    #                          title='Normalized Confusion matrix')
    #    
    #    fig_NormCM.savefig(model_summary_fname + ' - Norm Confusion Matrices.png',
    #                dpi=300,
    #                bbox_inches=0,
    #                facecolor=plotbgcolor,
    #                edgecolor='none',
    #                transparent=True)
    #    plt.close()

        del(confusionmatrix_data) # otherwise it keeps piling the results onto the previous one
    
    SimuCellChecker = (simucell_tracker != 0).any()
    
    if SimuCellChecker:
        confusion_out = np.concatenate((confusion_box,simucell_tracker), axis=1)
        headers_out = confusion_headers + ',' + simucell_headers
    else:
        confusion_out = confusion_box
        headers_out = confusion_headers
        
    confusion_fname = os.path.join(inputpath, 'AAA confusion summary.csv')
    np.savetxt(confusion_fname, confusion_out, delimiter=',', fmt='%10.3f', header=headers_out, comments='')


    # ========================================================================
    #    Plot Confusion Matrix for all files pooled
    # ========================================================================
    
    fn_etc.info_msg('Generating confusion matrices...')
    
    confusion_pool = np.delete(confusion_pool, (0), axis=0) # remove the first row
    
    # Training data confusion matrix
    confusionmatrix_trn = confusion_matrix(confusion_pool[:,0], confusion_pool[:,1])
    
    # Plot non-normalized confusion matrix
    fig_AbsCM = fn_etc.plot_confusion_matrix(confusionmatrix_trn, classes=['NC', 'C'],
                          cmap=cmap, plottxtcolor=plottxtcolor, plotbgcolor=plotbgcolor, normalize=False,
                          title='Absolute Confusion matrix')
    
    fig_AbsCM.savefig(os.path.join(inputpath, 'AAA Abs Confusion Matrices.png'),
                dpi=300,
                bbox_inches=0,
                facecolor=plotbgcolor,
                edgecolor='none',
                transparent=True)
    plt.close()
    
    # Plot normalized confusion matrix
    fig_NormCM = fn_etc.plot_confusion_matrix(confusionmatrix_trn, classes=['NC', 'C'],
                          cmap=cmap, plottxtcolor=plottxtcolor, plotbgcolor=plotbgcolor, normalize=True,
                          title='Normalized Confusion matrix')
    
    fig_NormCM.savefig(os.path.join(inputpath, 'AAA Norm Confusion Matrices.png'),
                dpi=300,
                bbox_inches=0,
                facecolor=plotbgcolor,
                edgecolor='none',
                transparent=True)
    plt.close()