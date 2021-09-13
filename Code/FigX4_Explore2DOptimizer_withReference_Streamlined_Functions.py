#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:46:41 2021

@author: andreas
"""

import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

def PlotScatter(XC_,labels=[],ax=[]):

    if(len(labels)==0):
        labels = -1*np.ones((len(XC_),));
 
    if(ax==[]):
        fig,ax = plt.subplots(1,1,figsize=(6,6)); 
       
    mark_ = (labels==-1);
    ax.scatter(x=XC_[mark_,0],y=XC_[mark_,1],s=.4,c='grey',alpha=0.1);

    mark_ = (labels>=0);
    sns.scatterplot(x=XC_[mark_,0],y=XC_[mark_,1],hue=labels[mark_],palette='deep',linewidth=0,
                    s=2,legend=False,ax=ax);
    ax.set_aspect('equal');
#    plt.savefig(outputfolder+"results_"+analysis_name+"_"+filename_add+".pdf",bbox_inches="tight");


def LoadPoints(filename,datascale=1):
    if(filename[-3:]=="txt"):
        XC = np.loadtxt(filename);
    elif(filename[-4:]=="hdf5"):
        f             = h5py.File(filename, 'r')    
        dset          = f['locs'];
        XC            = np.stack((dset["x"],dset["y"])).T    
        
    XC        = np.unique(XC,axis=0);
    XC        = datascale*XC;
        
    return XC;

def FilterPoints(XC,xyminmax):
    xmin,xmax,ymin,ymax = xyminmax[0],xyminmax[1],xyminmax[2],xyminmax[3];
    mask = (XC[:,0]>xmin)*(XC[:,0]<xmax)*(XC[:,1]>ymin)*(XC[:,1]<ymax);
    return XC[mask,:];


def GetLineOfOptima(df,x_selector,y_selector,no_bins=0):
    
    x_sel      = df[x_selector];
    x_sel_sort = np.sort(np.unique(x_sel));
    
    if(no_bins == 0):
        bins = np.asarray([np.min(x_sel)-1]+list((x_sel_sort[:-1]+x_sel_sort[1:])/2)+[np.max(x_sel)+1]);
        no_bins = len(bins)-1;
    else:
        bins = np.linspace(0.99*np.min(x_sel),np.max(x_sel)*1.01,no_bins+1);
    
    #print(np.unique(df[x_selector]));    

    xs = -1*np.ones((no_bins,1),dtype=int);
    idxs = [];
    
    for i in np.arange(no_bins):
        mark_    = (df[x_selector] > bins[i])&(df[x_selector] <= bins[i+1]);
        if(np.sum(mark_)==0):
            continue;
        else:
            idxs.append((df.loc[mark_,y_selector]).idxmax());
        
    df_opt             = pd.DataFrame();
    df_opt['idx']      = idxs;
    for c in df.columns:
        df_opt[c] = np.asarray(df.loc[idxs,c]);

    return df_opt;


def GetLineOfOptimaUnique(df,x_selector,y_selector,no_bins=0):
    
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
    for c in df.columns:
        df_opt[c] = np.asarray(df.loc[idxs,c]);

    return df_opt;


def GetClusterDistribution(labels):
    label_vs_cl_size = np.array([[l,np.sum(labels==l)] for l in np.unique(labels) if (l>=0)]);
    
#    cl_sizes = [];
#    for c in np.unique(labels):
#        if(c == -1):
#            continue;
#        cl_sizes.append(np.sum(labels==c));
    return label_vs_cl_size;


def GetClusterSizesAlongOptima(FD,df_opt_th):

    cl_dist    = [];
   # idxs       = [];
    thresholds = [];
    labels     = [];

    for index, row in df_opt_th.iterrows():
        df1_row     = FD.phasespace.loc[int(row['idx']),:];
        cld         = GetClusterDistribution(df1_row['labels']);
        cl_dist    += (list(cld[:,1]));
        labels     += (list(cld[:,0]));
    #    idxs       += list((int(row['idx']))*np.ones_like(cld[:,0]));
        thresholds += list(df1_row['threshold']*np.ones_like(cld[:,0]));

    df_clusterSizes = pd.DataFrame();
    df_clusterSizes['labels']      = labels;
    df_clusterSizes['clusterSize'] = cl_dist;
    df_clusterSizes['threshold']   = thresholds;
    
    return df_clusterSizes;

def AssembleStatistics(df_clusterSizes_):
    #data.groupby('month', as_index=False).agg({"duration": "sum"})
    rows_list = [];
    for t in np.unique(df_clusterSizes_['threshold']):

        d_ = df_clusterSizes_.loc[df_clusterSizes_['threshold']==t,'clusterSize'];
        
        dict1 = {}
        # get input row in dictionary format
        # key = col_name
        dict1['n']          = len(d_);
        dict1['mean']       = np.mean(d_);
        dict1['median']     = np.median(d_);
        dict1['quantile_90']    = np.quantile(d_,0.9);        
        dict1['quantile_10']    = np.quantile(d_,0.1);                
        dict1['cv']         = stats.variation(d_);
        dict1['threshold']  = t;
        dict1['fano']       = np.var(d_)/np.mean(d_);
        dict1['skewness']   = stats.skew(d_);
        dict1['kurtosis']   = stats.kurtosis(d_);

#        z_ = plt.hist(d_,bins=np.linspace(0,40,41)+0.5);
#        idx_max = np.argmax(z_[0]);    
#        max_cl.append(z_[1][idx_max]+0.5);
        #dict1['max_cl']   = max_cl;
        
        dv_ = d_.value_counts().sort_index();    
        v1  = np.sum(dv_[(dv_.index < t+1)]);#/np.sum(dv_[(dv_.index < t+3)]);
        dict1['firstBin']   = v1;

        rows_list.append(dict1)

    df_stats_per_th = pd.DataFrame(rows_list)               
    
    return df_stats_per_th;

def AnalyseClusterSizes(df_clusterSizes,df_clusterSizes_ref,filename):
    

    df_stats_per_th_signal = AssembleStatistics(df_clusterSizes);
    df_stats_per_th_signal['type'] = 'signal';
    
    df_stats_per_th_noise = AssembleStatistics(df_clusterSizes_ref);
    df_stats_per_th_noise['type'] = 'noise';    
    
    df_stats_per_th = pd.concat([df_stats_per_th_signal,df_stats_per_th_noise]);
    
    fig,axs = plt.subplots(3,2,figsize=(12,12)); 
    sns.lineplot(data=df_stats_per_th,x='threshold',y='cv',hue='type',ax=axs[0,0]);
    sns.lineplot(data=df_stats_per_th,x='threshold',y='fano',hue='type',ax=axs[0,1]);
    sns.lineplot(data=df_stats_per_th,x='threshold',y='skewness',hue='type',ax=axs[1,0]);
    sns.lineplot(data=df_stats_per_th,x='threshold',y='kurtosis',hue='type',ax=axs[1,1]);
    sns.lineplot(data=df_stats_per_th,x='threshold',y='median',hue='type',ax=axs[2,0]);
    sns.lineplot(data=df_stats_per_th,x='threshold',y='firstBin',hue='type',ax=axs[2,1]);    
    
    plt.savefig(filename,bbox_inches="tight");
    
    return df_stats_per_th;


def PlotDistribution(df,df_ref,filename):
    fig,axs = plt.subplots(1,2,figsize=(13,7)); 
    
    for i,df_ in enumerate([df,df_ref]):
            ax = axs[i];
            sns.stripplot(data=df_,y='clusterSize',x='threshold',ax=ax)
            sns.boxplot(data=df_,y='clusterSize',x='threshold',ax=ax,color='lightgrey')
            ax.set_ylim(0,80)
            ax.plot(np.arange(len(np.unique(df_['threshold']))),np.unique(df_['threshold']))
    
    plt.savefig(filename,bbox_inches="tight");