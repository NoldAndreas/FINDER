#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:29:27 2020

@author: andreas
"""
import pandas as pd
from AnalyseSeries import AnalyseSeries 
from AnalyseSeries import AnalyseSeries_PlotCases
import glob
from Basefolder import basefolder
import json

filenamesList = glob.glob(basefolder+'Results_Fig4/Results_*.txt')

for filename in filenamesList:
    #Load corresponding parameter file:
    filename_json = filename[:-4]+'_Parameters.json';
    with open(filename_json, 'r') as fp:
        params = json.load(fp)
            
    df = pd.read_csv(filename);       
    AnalyseSeries(df,params,filename[:-4]);

