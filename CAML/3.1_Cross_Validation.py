#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform k-fold cross-validation on existing models.

@author: dave

"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
import time
import keras
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame
from keras.utils import to_categorical

proc_wd = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print('Changed working directory to ' + proc_wd)

import FuncEtc as fn_etc
import FuncNormalizeInput as fn_normalize

# ========================================================================
#  Plotting variables
# ========================================================================
plotbgcolor = (0.1765,0.1765,0.1765)
plottxtcolor = (1,1,1)

ExportImageFormat = ['png'] # Format(s) to save images (given as a Python list).
                             # Format: 'eps', 'png', 'jpg', 'svg'
                             # Add multiple formats to the list and each image will
                             # be saved in that format. For example:                             
                             # PreviewImageFormat = ['eps','png'] saves EPS and PNG.


# ========================================================================
#  Cross-validation settings
# ========================================================================
CVSplits = 10              # how many subsets of data to be split from the input
                           # data. This is also how many rounds of CV to perform.
CVDataFrom = 'training'    # from the input PKL, which dataset to use
# CVDataFrom = 'validation' # you can use the other sets as well, if you wish!
# CVDataFrom = 'testing'

mdl_batchsize = 32
mdl_epochs = 100
mdl_learning_rate = 0.001
mdl_epoch_factor = 1.0    # influence of epochs on learning rate decay. Must be greater than zero!
mdl_decay_rate = mdl_learning_rate / (mdl_epoch_factor * mdl_epochs) # zero = no change in learning rate; it's always mdl_learning_rate for all epochs.

opt_adam = keras.optimizers.Adam(lr=mdl_learning_rate, decay=mdl_decay_rate)


# ========================================================================
#  End of user-editable variables
# ========================================================================

if __name__ == '__main__':
           
    todays_seed = 270879 # fix random seed for reproducibility

    modeljson = fn_etc.askforinput(
            message = 'Path to the JSON file describing the model configuration',
            errormessage= 'The file you provided does not exist or you have supplied a folder name instead of a file name.',
            defaultval= '',
            isvalid = lambda v : os.path.isfile(v))

    # extract some labels from the model filename
    model_json_fname = os.path.splitext(os.path.basename(os.path.normpath(modeljson)))[0]
    model_labels = model_json_fname.split()
    model_summary = ' '.join(model_labels[0:5]) 
    modelID = model_labels[0]
    DistancesProcessedAs = model_labels[2]

    # load the model
    # model = load_model(modelfile, custom_objects= {'f1_score': fn_etc.f1_score})
    if os.path.exists(modeljson):    
        with open(modeljson,'r') as json_file:
            loaded_model_json = json_file.read()
        model_home_folder = os.path.dirname(modeljson)
        timestamp = time.strftime("%y%m%d_%H%M", time.localtime())
        outputfolder = os.path.join(model_home_folder, timestamp + ' - ' + str(CVSplits) + '-fold Cross-Validation')
        
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
    else:
        raise ValueError('You must supply a valid model JSON file.')

    # Does training data exist from previous 1_DataPreparation script?
    Load_PKL = True
    if 'TrainValTest_data_fname' in locals() or 'inputpkl' in locals():
        
        try:
            previousdata = TrainValTest_data_fname
        except:
            previousdata = inputpkl

        if 'recall_data' in locals():
            recycleRecallData = fn_etc.askforinput(
                message = 'Do you want to re-use the raw (distances) training, validation, and testing data (loaded from ' + previousdata + ')?',
                errormessage = 'Please enter y or n',
                defaultval = '',
                isvalid = lambda v : v.lower() in ['y','n','yes','no'])
            
            if recycleRecallData.lower() in ['y','yes']:
                fn_etc.ok_msg('Using the existing training, validation, and testing data.')
                Load_PKL = False

    if Load_PKL:
        # We are loading a file but we can still suggest the last-used location as default.
        if 'TrainValTest_data_fname' in locals() and os.path.exists(TrainValTest_data_fname):
            default_input_pkl = TrainValTest_data_fname
        elif 'inputpkl' in locals() and os.path.exists(inputpkl):
            default_input_pkl = inputpkl
        else:
            default_input_pkl = ''
            
        # get the input data for validation
        inputpkl = fn_etc.askforinput(
            message = 'Path to the .PKL file containing pooled training, validation, and testing data',
            errormessage = 'The file you provided does not exist or you have given a folder name instead of a file name.',
            defaultval = default_input_pkl,
            isvalid = lambda v : os.path.isfile(v))
        
        with open(inputpkl, 'rb') as f:
            recall_data = pickle.load(f, encoding='latin1')
        
    # inputfolder = os.path.dirname(inputpkl)

    # ========================================================================
    #  Model Preparation: Turn raw input (distances) into input for the model
    # ========================================================================
    if 'X_training_data' and 'X_validation_data' and 'X_testing_data' in locals() and not Load_PKL:
        recyclePreparedData = fn_etc.askforinput(
            message = 'Existing prepared data (by ' + PreparationType + ' method) for training, validation, and testing exists. Do you want to re-use it?',
            errormessage = 'Please enter y or n',
            defaultval = default_input_pkl,
            isvalid = lambda v : v.lower() in ['y','n','yes','no'])
        
        if recyclePreparedData.lower() in ['n','no']:
            RedoDataPrep = True
        else:
            RedoDataPrep = False
            fn_etc.ok_msg('Training, validation, and testing data will be re-used. NB: These data were prepared as \'' + PreparationType + '\'')
            
    else:
        RedoDataPrep = True

    if RedoDataPrep:
        
        # ========================================================================
        #  Import training, validation, and testing data
        # ========================================================================
        #
        ### Load existing training, validation, and testing data pool
    
        print('Importing PKL file...')
        
        # load items from our pickle and create variables based on the variable name
        # stored in the first pickled item (the list of variable names)
        pickle_open_file = open(inputpkl, 'rb' ) # open the file for reading
        load_order = pickle.load(pickle_open_file) # retrieve the first item, which is the list in which everything was saved
        for item in load_order:
            locals()[item] = pickle.load(pickle_open_file) # load each item and convert it into a local variable named according to 'item'
        pickle_open_file.close()
        
        
        # The following variables will already have been loaded.
        loaded_pickle = True # this is to avoid dropping into the if statement below
        if not loaded_pickle:
            total_training_count = None          # These variables have been loaded
            training_label_fraction = None       # from the pickle file, above.
            training_reqd_counts = None          #
            X_training_distances_raw = None      # Normally we have to recall the
            Y_training_vectors_raw = None        # names in the exact same order
            Y_training_binary_raw = None         # they were saved but here we used
            Y_training_labels_raw = None         # a list (loaded from the same
            total_validation_count = None        # pickle file) to know the names
            validation_label_fraction = None # and the order they were saved.
            validation_reqd_counts = None        #
            X_validation_distances_raw = None    # In case you are trying to search
            Y_validation_vectors_raw = None      # for the names to find where they
            Y_validation_binary_raw = None       # originate they are stated here
            Y_validation_labels_raw = None       # explicitly but in an 'if' which
            total_testing_count = None           # should never be executed.
            testing_label_fraction = None    #
            testing_reqd_counts = None           # Their explict statement here
            X_testing_distances_raw = None       # should also help remove syntax
            Y_testing_vectors_raw = None         # errors in your IDE.
            Y_testing_binary_raw = None          # 
            Y_testing_labels_raw = None          # To change these variable names
            ps = None                            # edit the script for the previous
            label_names = None                   # Stage 2 (pooling training data)

        fn_etc.ok_msg('Training, validation, and testing data were unpickled from ' + inputpkl)


        # Extract the original training variables from the recalled data:
        if CVDataFrom == 'training':
    
            total_CrossValidation_count = total_training_count
            X_CrossValidation_raw = X_training_distances_raw
            Y_CrossValidation = Y_training_labels_raw
    
        elif CVDataFrom == 'validation':
        
            total_CrossValidation_count = total_validation_count
            X_CrossValidation_raw = X_validation_distances_raw
            Y_CrossValidation = Y_validation_labels_raw
    
        elif CVDataFrom == 'testing':
        
            total_CrossValidation_count = total_testing_count
            X_CrossValidation_raw = X_testing_distances_raw
            Y_CrossValidation = Y_testing_labels_raw
            
        else:
            fn_etc.err_msg('Can\'t work out which data to use for Cross Validation!')
            raise ValueError('Cross Validation data must be from either the Training, Validation, or Testing dataset. Check \'CVDataFrom\' variable...')
    
        fn_etc.ok_msg('Cross Validation data (using the ' + CVDataFrom + ' dataset) was unpickled from ' + inputpkl)
    
        # Prepare the CrossValidation data for the model
        X_CrossValidation = fn_normalize.normalize_dists(X_CrossValidation_raw, total_CrossValidation_count, DistancesProcessedAs)    
    
#        # reshape data to be repeats/measures/features for LSTM
#        X_CrossValidation = X_CrossValidation.reshape((total_CrossValidation_count, ps['FurthestFriend'], 1))
#        
#        if X_training_data.ndim != 3:
#            X_training_data = X_training_data.reshape((total_training_count, ps['FurthestFriend'], mdl_features_types))

    else:
        
        # Reuse existing data
        if CVDataFrom == 'training':
    
            total_CrossValidation_count = total_training_count
            X_CrossValidation = X_training_data
            Y_CrossValidation = Y_training_data
    
        elif CVDataFrom == 'validation':
        
            total_CrossValidation_count = total_validation_count
            X_CrossValidation = X_validation_data
            Y_CrossValidation = Y_validation_data
    
        elif CVDataFrom == 'testing':
        
            total_CrossValidation_count = total_testing_count
            X_CrossValidation = X_testing_data
            Y_CrossValidation = Y_testing_data
            
        else:
            fn_etc.err_msg('Can\'t work out which data to use for Cross Validation!')
            raise ValueError('Cross Validation data must be from either the Training, Validation, or Testing dataset. Check \'CVDataFrom\' variable...')

        

    # ========================================================================
    #    Ten-fold cross validation
    # ========================================================================

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=CVSplits, shuffle=True, random_state=todays_seed)
    cvscores = []

    CV_train_accu = DataFrame()
    CV_train_loss = DataFrame()
    CV_iteration_tracker = 0

    for train, validate in kfold.split(X_CrossValidation, Y_CrossValidation):

        fn_etc.progress_msg(str(CV_iteration_tracker + 1) + ' of ' + str(CVSplits), modelID + ' ' + str(CVSplits) + '-fold Cross-Validation (' + str(mdl_epochs) + ' epochs)')

        modelCV = None # Wipe out any existing (trained) versions of this model.
        modelCV = keras.models.model_from_json(loaded_model_json) # Build model fresh from JSON
        
        training_X = X_CrossValidation[train]
        validate_X = X_CrossValidation[validate]
        
        if len(label_names) == 2:
            modelCV.compile(loss='binary_crossentropy', optimizer=opt_adam, metrics=['accuracy'])
            training_Y = Y_CrossValidation[train]
            validate_Y = Y_CrossValidation[validate]
        else:
            modelCV.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy'])
            # convert our integer training labels into binary encoded
            training_Y = to_categorical(Y_CrossValidation[train])
            validate_Y = to_categorical(Y_CrossValidation[validate])

        #Fit the model
        CV_history = modelCV.fit(training_X, training_Y, epochs=mdl_epochs, batch_size=mdl_batchsize, verbose=3)

        # Log the history for this round of training and validation
        CV_train_accu['Round ' + str(CV_iteration_tracker + 1)] = CV_history.history['acc']
        CV_train_loss['Round ' + str(CV_iteration_tracker + 1)] = CV_history.history['loss']

        scores = modelCV.evaluate(validate_X, validate_Y, verbose=3)
        cvscores.append(scores[1] * 100)

        print('Finished round %i with Accuracy: %.2f%% and Loss: %.4f' % (CV_iteration_tracker + 1, scores[1]*100, scores[0]))

        CV_iteration_tracker += 1 # increment the tracker
        del CV_history # also flush this before the next round
        
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    # plot the CV training and validation accuracy over epochs
    fig_CVTrnVal_Accu = plt.figure(facecolor = plotbgcolor)
    ax_CVTrnVal_Accu = fig_CVTrnVal_Accu.add_subplot(111, facecolor = plotbgcolor)
    fig_CVTrnVal_Accu.set_size_inches(10, 10)
    
    plt.plot(CV_train_accu)
    plt.ylim((0.5,1.0))
    # fix axis colours
    ax_CVTrnVal_Accu.tick_params(color=plottxtcolor, labelcolor=plottxtcolor)
    for spine in ax_CVTrnVal_Accu.spines.values():
        spine.set_edgecolor(plottxtcolor)
    # add legend and match colours
    leg = plt.legend(CV_train_accu, loc='lower right', facecolor = plotbgcolor)
    for handle, text in zip(leg.legendHandles, leg.get_texts()):
        text.set_color(handle.get_color())
    # add labels etc
    plt.title('Model ' + modelID + ' ' + str(CVSplits) + '-fold Cross Validation Accuracy', color=plottxtcolor)
    plt.ylabel('Accuracy', color=plottxtcolor)
    plt.xlabel('Epoch', color=plottxtcolor)    
    fig_CVTrnVal_Accu.tight_layout()
    plt.show()
    # save it
    for ExportImage in ExportImageFormat:
        fname = os.path.join(outputfolder, model_summary + ' - ' + str(CVSplits) + '-fold CV Accuracy Plot (' + str(mdl_epochs) + ' epochs on ' + CVDataFrom + ' pts).' + ExportImage)
        fig_CVTrnVal_Accu.savefig(fname,
                                  dpi=300,
                                  bbox_inches=0,
                                  facecolor=plotbgcolor,
                                  edgecolor='none',
                                  transparent=True)
    plt.close()

    # plot the CV training and validation loss over epochs
    fig_CVTrnVal_Loss = plt.figure(facecolor = plotbgcolor)
    ax_CVTrnVal_Loss = fig_CVTrnVal_Loss.add_subplot(111, facecolor = plotbgcolor)
    fig_CVTrnVal_Loss.set_size_inches(10, 10)
    
    plt.plot(CV_train_loss)
    loss_y_max = np.ceil(np.max(CV_train_loss.max(axis=0)) * 10) / 10
    plt.ylim((0, np.max((loss_y_max, 0.5))))
    #plt.plot(CV_val_loss.history['val_loss'], color='cyan')
    # fix axis colours
    ax_CVTrnVal_Loss.tick_params(color=plottxtcolor, labelcolor=plottxtcolor)
    for spine in ax_CVTrnVal_Loss.spines.values():
        spine.set_edgecolor(plottxtcolor)
    # add legend and match colours
    leg = plt.legend(CV_train_loss, loc='upper right', facecolor = plotbgcolor)
    for handle, text in zip(leg.legendHandles, leg.get_texts()):
        text.set_color(handle.get_color())
    # add labels etc
    plt.title('Model ' + modelID + ' ' + str(CVSplits) + '-fold Cross Validation Loss', color=plottxtcolor)
    plt.ylabel('Loss', color=plottxtcolor)
    plt.xlabel('Epoch', color=plottxtcolor)    
    fig_CVTrnVal_Loss.tight_layout()
    plt.show()
    # save it
    for ExportImage in ExportImageFormat:
        fname = os.path.join(outputfolder, model_summary + ' - ' + str(CVSplits) + '-fold CV Loss Plot (' + str(mdl_epochs) + ' epochs on ' + CVDataFrom + ' pts).' + ExportImage)
        fig_CVTrnVal_Loss.savefig(fname,
                    dpi=300,
                    bbox_inches=0,
                    facecolor=plotbgcolor,
                    edgecolor='none',
                    transparent=True)
    plt.close()

    # export CVS scores to text file
    fname = os.path.join(outputfolder, model_summary + ' - ' + str(CVSplits) + '-fold Cross Validation (' + str(mdl_epochs) + ' epochs on ' + CVDataFrom + ' pts).txt')
    with open(fname,'a') as cvstats:
        writer = csv.writer(cvstats, delimiter='\t')
        writer.writerow(['------------------------------------------------------'])
        writer.writerow([model_summary])
        writer.writerow(['------------------------------------------------------'])
        writer.writerow(["%.2f%% ± %.2f%%" % (np.mean(cvscores), np.std(cvscores))])
        writer.writerow(['------------------------------------------------------'])
        for cv_item in cvscores:
            writer.writerow([cv_item])
        writer.writerow([''])