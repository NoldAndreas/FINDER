#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import os, shutil

def Clustering_CAML(algo,XC,file_txt='default.tsv',datafolder=[]):    

    print(os.getcwd())
    print(datafolder);
    datafolder = datafolder + 'CAML/';
    #*********************
    #Delete all content in folder    
    for filename in os.listdir(datafolder + "Output/"):
        file_path = os.path.join(datafolder + "Output/", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    
    global input_PrepJSON
    global output_folder;
    global s1_prep_outputpath
    global model_fname
    global files
    
    output_folder = datafolder + "Output/";
    np.savetxt(datafolder + "Output/" + file_txt,XC,delimiter=' \t',fmt='%4.4f',header="x \t y",comments='');

    #Get current working directory
    cwd = os.getcwd()
    
    if('87B144' in algo):        
        model_ = '87B144';
    elif('07VEJJ' in algo):        
        model_ = '07VEJJ';        
    
    if(model_ in ["87B144"]):
        input_PrepJSON = datafolder  + "Input/" + 'AAA - MyData_1000.json';
    elif(model_ in ["07VEJJ"]):
        input_PrepJSON = datafolder + "Input/"  + 'AAA - MyData_100.json';
        
    
    if(model_ == "87B144"):
        s1_prep_outputpath = datafolder +'1_prep_dNN(1-1000)_novel';
        model_fname        = datafolder + 'CAML_TrainedModels/2D/12 Layers/87B144 (1D, 1000 NN)/87B144 - Norm-Self Train(500.0k,0.5×Clus) Val(100.0k,0.5×Clus).h5';                                                                                                
    elif(model_ == "07VEJJ"):    
        s1_prep_outputpath = datafolder +'1_prep_dNN(1-100)_novel';
        model_fname        = datafolder + 'CAML_TrainedModels/2D/12 Layers/07VEJJ (2D, 100 NN)/07VEJJ - Norm-Self Train(500.0k,0.5×Clus) Val(100.0k,0.5×Clus).h5';       
    
    if(True):
        try:
            with open("CAML/1.0_Data_Preparation.py") as f:        
                code = compile(f.read(), "CAML/1.0_Data_Preparation.py", 'exec')
                exec(code,globals())
        except:
            print("Execution of CAML/1.0_Data_Preparation.py halted");
            success = False

            #Change back to original working directory
        #if os.getcwd() != cwd:
        #    os.chdir(cwd)
        #    print('Changed working directory back to ' + cwd)
        
        
    if(True):   
        try:
            with open("CAML/4.0_Evaluation_With_ClusterClub.py") as f:         
                code = compile(f.read(), "CAML/4.0_Evaluation_With_ClusterClub.py", 'exec')
                exec(code,globals());
            print("Successful execution of CAML/4.0_Evaluation_With_ClusterClub.py");
            success = True;
        except:
            print("Execution of CAML/4.0_Evaluation_With_ClusterClub.py halted");
            success = False;
            
    if(success):
        all_     = np.loadtxt(datafolder + "Output/" + '4_evaluated_by_'+model_+'/'+file_txt[:-4]+'_DataCalled.tsv',skiprows=1);
        labels_  = (all_[:,6]-1).astype('int16');
        XC       = all_[:,0:2];    
    else:
        labels_  = -np.ones((len(XC),),dtype='int16');


    #Change back to original working directory
    if os.getcwd() != cwd:
        os.chdir(cwd)
        print('Changed working directory back to ' + cwd)
    
    return labels_



    
    
    