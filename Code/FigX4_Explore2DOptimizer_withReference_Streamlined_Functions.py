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

def RemoveLabelsSmallerT(labels_input,df,T,criterion):
    labels           = labels_input.copy();
    labels_to_remove = df.loc[df[criterion]<T,'labels'];
    for l in np.unique(labels_to_remove):
        labels[labels == l] = -1;
    return labels;

def GetOptimalT(df_incell,df_outcell,criterion,min_noClustersToAnalyse=10,bestRequiredRate=1.0):

    #ts = np.linspace(10,df_incell['clusterSize'].max(),500);
    ts = [];
    incell_vs_outcell_probability = [];
    for t in np.unique(df_incell[criterion]):
        no_clusters_over_t_outcell  = np.sum(df_outcell[criterion]>t)/len(df_outcell[criterion]);
        no_clusters_over_t_incell   = np.sum(df_incell[criterion]>t)/len(df_incell[criterion]);
        if(np.sum(df_outcell[criterion]>t)+ np.sum(df_incell[criterion]>t)>min_noClustersToAnalyse):
            incell_vs_outcell_probability.append(no_clusters_over_t_incell/(no_clusters_over_t_outcell+no_clusters_over_t_incell));
            ts.append(t);


    if(len(incell_vs_outcell_probability)==0):
        T = 0;
        incell_vs_outcell_probability_max = 0;
        no_clusters_incell = 0;
        no_clusters_outcell = 0;
    else:
        h = np.minimum(incell_vs_outcell_probability,bestRequiredRate);
        print(bestRequiredRate)
        incell_vs_outcell_probability_max = np.max(h);
        imax = np.argmax(h);
        T    = ts[imax];
        incell_vs_outcell_probability_max = incell_vs_outcell_probability[imax];

        no_clusters_incell              = np.sum(df_incell[criterion]>T);
        no_clusters_outcell               = np.sum(df_outcell[criterion]>T);

    plt.plot(ts,incell_vs_outcell_probability);
    plt.xlim(0,2*T)
    return T,incell_vs_outcell_probability_max,no_clusters_incell,no_clusters_outcell;

def DefineCleanedLabels_GeneralLimit(df_clusterSizes_all,phasespace_all,criterion,bestRequiredRate):
    #
#    criterion = 'clusterSize';

    labels_incell = [];
    no_cl_incell  = [];
    labels_outcell  = [];
    no_cl_outcell   = [];
    T = [];
    s_vs_n = [];
    percent_locsIncluded_aboveT     = [];
    percent_locsIncluded_aboveT_ref = [];

    df_incell   = df_clusterSizes_all[(df_clusterSizes_all['type']=='incell')];
    df_outcell  = df_clusterSizes_all[(df_clusterSizes_all['type']=='outcell')];

    T_,s_vs_n_,nocl_incell_,nocl_outcell_ = GetOptimalT(df_incell,df_outcell,criterion,bestRequiredRate=bestRequiredRate);
    print('T= '+str(T_));
#    T_ = 190;

    for idx,row in phasespace_all.iterrows():
        th,si = row['threshold'],row['sigma'];

        df_incell = df_clusterSizes_all[(df_clusterSizes_all['threshold']==th)&\
                        (df_clusterSizes_all['sigma']==si)&\
                        (df_clusterSizes_all['type']=='incell')];
        df_outcell  = df_clusterSizes_all[(df_clusterSizes_all['threshold']==th)&\
                        (df_clusterSizes_all['sigma']==si)&\
                        (df_clusterSizes_all['type']=='outcell')];

       # T_,s_vs_n_,nocl_incell_,nocl_outcell_ = GetOptimalT(df_incell,df_outcell,criterion);
        nocl_incell_  = float(np.sum(df_incell[criterion]>T_));
        nocl_outcell_ = float(np.sum(df_outcell[criterion]>T_));
        if(nocl_incell_ + nocl_outcell_>0):
            p1 = (nocl_incell_/len(df_incell));
            p2 = (nocl_outcell_/len(df_outcell));
            s_vs_n_        = p1/(p1+p2);
        else:
            s_vs_n_ = 0;

        no_cl_incell.append(nocl_incell_);
        no_cl_outcell.append(nocl_outcell_);
        T.append(T_);
        s_vs_n.append(s_vs_n_);

        #*******************************************************
        # Get labels of clusters larger than T
        l_ = RemoveLabelsSmallerT(row['labels'],df_incell,T_,criterion)
        labels_incell.append(l_);
        percent_locsIncluded_aboveT.append(np.sum(l_>=0)/len(l_))

        l_ = RemoveLabelsSmallerT(row['labels_ref'],df_outcell,T_,criterion)
        labels_outcell.append(l_);
        percent_locsIncluded_aboveT_ref.append(np.sum(l_>=0)/len(l_))
        #*******************************************************

    phasespace_all_aboveT = pd.DataFrame();
    phasespace_all_aboveT['sigma'] = phasespace_all['sigma'];
    phasespace_all_aboveT['threshold'] = phasespace_all['threshold'];
    phasespace_all_aboveT['labels']     = labels_incell;
    phasespace_all_aboveT['labels_ref'] = labels_outcell;

    phasespace_all_aboveT['no_clusters']     = no_cl_incell;
    phasespace_all_aboveT['no_clusters_ref'] = no_cl_outcell;

    phasespace_all_aboveT['no_clusters_s_vs_n'] = s_vs_n;
    phasespace_all_aboveT['T'] =     T;

    phasespace_all_aboveT['percent_locsIncluded']     = percent_locsIncluded_aboveT;
    phasespace_all_aboveT['percent_locsIncluded_ref'] = percent_locsIncluded_aboveT_ref;

    return phasespace_all_aboveT;

def GetDensity(XC):
    x_minmax = (np.max(XC,axis=0))-np.min(XC,axis=0);
    area     = x_minmax[0]*x_minmax[1];
    return len(XC)/area;

def GetOverlay(XC_incell,XC_outcell):

    if(('overlay_outcell' in parameters.keys()) and (parameters['overlay_outcell']==1)):
        density_ration_inoutcell = GetDensity(XC_incell)/GetDensity(XC_outcell);

        n_x = int(np.round(np.sqrt(density_ration_inoutcell)));
        n_y = int(np.round(density_ration_inoutcell/n_x));
        print('Deviding in '+str(n_x)+' x '+str(n_y))

        XC_outcell_overlay = np.zeros([0,2])
        xis = np.linspace(np.min(XC_outcell[:,0]),np.max(XC_outcell[:,0]),n_x+1);
        yis = np.linspace(np.min(XC_outcell[:,1]),np.max(XC_outcell[:,1]),n_y+1);
        for i,xl in enumerate(xis[:-1]):
            xr   = xis[i+1];

            for i,yl in enumerate(yis[:-1]):
                yr   = yis[i+1];
                mark = (XC_outcell[:,0]>=xl)&(XC_outcell[:,0]<=xr);
                mark = mark&(XC_outcell[:,1]>=yl)&(XC_outcell[:,1]<=yr);
                XC_paste = XC_outcell[mark,:]-[xl,yl];
                if(np.random.rand()>0.5):
                    XC_paste = XC_paste[:,[1,0]];
                XC_outcell_overlay = np.concatenate((XC_outcell_overlay,XC_paste))
               # break;
        #    break;
        #    for y_i np.linspace(np.min(XC_outcell[:,0]),np.max(XC_outcell[:,0]),5):
#        XC_outcell = XC_outcell_overlay;
        print('Ratio of localization densities: '+str(GetDensity(XC_incell)/GetDensity(XC_outcell)));
    else:
        XC_outcell_overlay = XC_outcell;
        print('No overlay for outcell');
    return XC_outcell_overlay;

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

    print(str(len(XC)) + ' points loaded from '+filename);
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

#def GetClusterSizesAll(FD):#

#    cl_dist    = [];
#    thresholds = [];
#    sigmas     = [];
#    labels     = [];
#
#    for index, df1_row in FD.phasespace.iterrows():
#        #df1_row     = FD.phasespace.loc[int(row['idx']),:];
#        cld         = GetClusterDistribution(df1_row['labels']);
##        print(cld.shape)
#        if(cld.shape[0] == 0):
#            continue;
#        cl_dist    += (list(cld[:,1]));
#        labels     += (list(cld[:,0]));
#        thresholds += list(df1_row['threshold']*np.ones_like(cld[:,0]));
#        sigmas     += list(df1_row['sigma']*np.ones_like(cld[:,0]));#
#
#    df_clusterSizes = pd.DataFrame();
#    df_clusterSizes['labels']      = labels;
#    df_clusterSizes['clusterSize'] = cl_dist;
#    df_clusterSizes['threshold']   = thresholds;
#    df_clusterSizes['sigma']       = sigmas;
#
#    return df_clusterSizes;

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
