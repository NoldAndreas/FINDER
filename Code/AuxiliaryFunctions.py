#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:18:32 2021

@author: andreas
"""
import numpy as np
import pandas as pd

def GetLineOfOptima(df,x_selector,y_selector,includeAllColumns=False):
    
    x_sel_sort = np.sort(np.unique(df[x_selector]));    
    idxs       = [];
    
    for x_sel in x_sel_sort:
        mark_    = (df[x_selector] == x_sel);
        if(np.sum(mark_)==0):
            continue;
        else:
            idxs.append((df.loc[mark_,y_selector]).idxmax());
        
    df_opt             = pd.DataFrame();
    df_opt['idx']      = idxs;
    
    if(includeAllColumns):
        cols = df.columns;
    else:
        cols = [x_selector,y_selector];

    for c in cols:
        df_opt[c] = np.asarray(df.loc[idxs,c]);   

        
    return df_opt;


def GetClusterDistribution(labels):
    cl_sizes = [];
    for c in np.unique(labels):
        if(c == -1):
            continue;
        cl_sizes.append(np.sum(labels==c));
    return cl_sizes;