#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

sys.path.append("Modules/")

import os
import time
import numpy as np
import matplotlib.pyplot as plt
# import scipy.spatial.distance as dist
import seaborn as sns
from sklearn.cluster import DBSCAN

import h5py
from sklearn.neighbors import NearestNeighbors
# from astropy.stats import RipleysKEstimator
# from varname import nameof

from Finder import Finder
from Clustering_CAML import Clustering_CAML
from DbscanLoop import DbscanLoop

from Clustering import Clustering
from Definitions import get_datafolder


# FUNCTIONS

def PlotDistribution(labels, ax, col):
    cluster_sizes = [];
    for i in np.unique(labels):
        if (i == -1):
            continue;
        cluster_sizes.append(np.sum(labels == i));
    cluster_sizes = np.asarray(cluster_sizes);
    locs_in_large_clusters = np.sum(cluster_sizes[cluster_sizes > edges_max_clusterSizeDist]);

    mark_ = (cluster_sizes > edges_max_clusterSizeDist);
    print("Noise localizations: " + str(np.sum(labels == -1)) + " or " + str(
        np.round(100 * np.sum(labels == -1) / len(labels), 2)) + " %");
    print("Clusters with size >= " + str(edges_max_clusterSizeDist) + ": " + str(
        np.sum((cluster_sizes > edges_max_clusterSizeDist))));
    print("Number of localzations in clusters with size >= " + str(edges_max_clusterSizeDist) + ' : ' + str(
        locs_in_large_clusters) +
          ' or ' + str(100 * locs_in_large_clusters / len(labels)) + ' %');
    print(cluster_sizes[mark_]);
    cluster_sizes[mark_] = edges_max_clusterSizeDist - 1;
    edges = np.linspace(0, edges_max_clusterSizeDist, 41);

    #    sns.histplot(cluster_sizes,ax=ax,kde=True,bins=edges,shrink=0.8,color=col);
    sns.histplot(cluster_sizes, ax=ax, kde=True, shrink=0.8, color=col, bins=edges, );
    ax.set_xlabel('Cluster size (localizations)');
    ax.set_frame_on(False);

    ax.set_ylim(0, y_max_clusterSizeDist);
    ax.set_xticks(xticks_clusterSizeDist);
    # ax.set_yticks(yticks);
    xlabs = [str(i) for i in xticks_clusterSizeDist];
    xlabs[-1] = "\u2265 " + xlabs[-1];
    ax.set_xticklabels(xlabs)


def PlotScatter(labels, XC, markPlot, ax=[], ax_dist=[], col='grey', showScaleBar=False, showBorder=False,
                highlights=[]):
    cols = sns.color_palette("deep", 10)

    XCplot = XC[markPlot, :];
    labelsPlot = labels[markPlot];

    # Get correctly detected:
    mark = (labelsPlot == -1);
    ax.scatter(x=XCplot[mark, 0], y=XCplot[mark, 1], s=.4, c='grey', alpha=0.1);
    #    sns.scatterplot(x=XC[mark,0],y=XC[mark,1],linewidth=0,color='grey',alpha=0.2,ax=ax,legend=False);
    mark = (labelsPlot >= 0);
    # ax.scatter(x=XC[mark,0],y=XC[mark,1],s=.4,c=labels[mark],cmap='tab20');#,c=cols);
    #   ax.scatter(x=XC[mark,0],y=XC[mark,1],marker='o',s=1,c=cols,alpha=0.1,edgecolor=None,linewidth=0);
    sns.scatterplot(x=XCplot[mark, 0], y=XCplot[mark, 1], hue=labelsPlot[mark]
                   , palette='deep'
                    , linewidth=0,
                    s=2, legend=False, ax=ax);
    ax.set_aspect('equal');

    if (case == 'neuron'):
        dy = 300;
        dx = 1000;
        str_ = '$1\mu m$';
    elif (case == 'protein_ttx_1hr_2'):
        dy = 1000;
        dx = 2000;
        str_ = '$2\mu m$';

    x_0 = np.min(XCplot[:, 0]) + dx / 4;
    y_0 = np.min(XCplot[:, 1]) - dy;

    if (showScaleBar):
        ax.plot([x_0, x_0 + dx], [y_0, y_0], 'k')
        ax.annotate(str_, (x_0 + dx / 2, y_0 + dy / 3), ha='center');  # fontsize='normal'
    else:
        ax.plot([x_0, x_0 + 10], [y_0, y_0], 'w')

    labs = ['I', 'II', 'III', 'IV', 'V'];
    for i_, highlight in enumerate(highlights):
        circle_ = plt.Circle(highlight["xy"], highlight["R"], color='r', fill=False)
        ax.add_patch(circle_);
        pos_ = np.asarray(highlight["xy"]) + highlight["R"] * 0.8 * np.asarray([1, 1]);
        ax.annotate(labs[i_], pos_, color='r', )

    ax.set_aspect(1);
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.axis('off');

    if (not (ax_dist == [])):
        PlotDistribution(labels, ax_dist, col);

    if (ax == []):
        plt.show();


#        plt.savefig(filename)

def PlotImageNoNoise(XC, labels, name):
    labels_large = -1 * np.ones_like(labels);
    for i in np.unique(labels):
        if (i == -1):
            continue;
        if (np.sum(labels == i) > edges_max_clusterSizeDist):
            labels_large[labels == i] = i;

    fig, ax = plt.subplots(1, 1, figsize=(5, 5));
    #   mark   = (labels_large>=0);
    ax.scatter(XC[labels >= 0, 0], XC[labels >= 0, 1], 0.05, c='k', alpha=0.1);
    ax.scatter(XC[labels_large >= 0, 0], XC[labels_large >= 0, 1], 0.05, c='r', alpha=0.1);
    #    sns.scatterplot(x=XC[mark,0],y=XC[mark,1],hue=labels[mark],palette='deep',linewidth=0,
    #                    s=2,legend=False,ax=ax,alpha=0.1);

    ax.set_title(name);
    dx = 10000;
    str_ = '$10\mu m$';
    x_0 = np.min(XC[:, 0]) + 5000;
    y_0 = np.min(XC[:, 1]) - 2500;
    ax.plot([x_0, x_0 + dx], [y_0, y_0], 'k')
    ax.annotate(str_, (x_0 + dx / 2, y_0 + 1000), ha='center');  # fontsize='normal

    ax.set_aspect(1);
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.axis('off');

    plt.savefig(results_folder + "/" + case + "_overview_" + name + "_noNoise.png", dpi=300, bbox_inches="tight");


# END FUNCTION


# ---------------------------------------------------------------------------------------------------------------------


basefolder = get_datafolder()

# base_folder_2 = os.path.dirname(os.getcwd())
# data_folder = os.path.join(os.path.dirname(os.getcwd()), 'data_sources')


my_pal = {'FINDER_loop': '#701ac0',
          'CAML_07VEJJ': '#eabe8e',
          'CAML_87B144': '#d67d1d',
          'FINDER': '#af6eeb',
          'dbscan': 'dimgrey',
          'OPTICS': 'lightgrey'
          }

results_folder = os.path.join(basefolder, "Results_Fig5")
case = 'neuron' #'neuron';  # 'neuron';#'protein_ttx_1hr_2';
computeClustering = True
plotFullImage = True
plot10PtDistance = False
plotFullImageNoNoise = True
plotMainAnalysis = True

# 1 unit corresponds to 158nm. We normalize such that 1 unit = 1 nanometer
datascale = 158;  # =81/512*1000

threshold = 10;  # for dbscan

highlights = [];
if (case == 'neuron'):
    filename = '/TemplateClusters/NeuronData/dendrite_example_Cell1_GluA2_40I_ROI1_1_MMStack_Pos0.ome_locs_render_driftCorr_filter_render_pix.6fr20_picked2_picked3.txt';
    edges_max_clusterSizeDist = 200;
    y_max_clusterSizeDist = 20;
    xticks_clusterSizeDist = [0, 50, 100, 150, 200];
    lims_x = [-np.Inf, np.Inf];
    lims_y = [-np.Inf, np.Inf];

elif ((case == 'ribosome_40s')):
    filename = 'TemplateClusters/RibosomeData/40s.txt';
    edges_max_clusterSizeDist = 1200;
    xticks_clusterSizeDist = [0, 400, 800, 1200];
    lims_x = [158 * 350, 158 * 400];
    lims_y = [158 * 250, 158 * 300];
elif ((case == 'ribosome_60s')):
    edges_max_clusterSizeDist = 1200;
    y_max_clusterSizeDist = 100;
    xticks_clusterSizeDist = [0, 400, 800, 1200];
    lims_x = [158 * 350, 158 * 400];
    lims_y = [158 * 250, 158 * 300];
    filename = 'TemplateClusters/RibosomeData/60s.txt';

elif (case == 'protein_ttx_1hr_2'):
    print(case)
    edges_max_clusterSizeDist = 400;
    y_max_clusterSizeDist = 3500;
    xticks_clusterSizeDist = [0, 100, 200, 300, 400];
    #    lims_x = [158*210,158*280];
    #    lims_y = [158*25,158*85];
    lims_x = [37000, 43000];
    lims_y = [4000, 10000];

    filename = 'TemplateClusters/ProteinData_ttx_1hr_2/AHA_2_MMStack_Pos0.ome_locs_render_driftcor_filter_render_pix0.02X6f20.hdf5'
    # highlights.append({"xy":[2250+158*210,7000+158*30-500],"R":550});
    highlights.append({"xy": [39900, 9400], "R": 600});
    highlights.append({"xy": [6070 + 158 * 210, 580 + 158 * 30 - 500], "R": 600});
    highlights.append({"xy": [8200 + 158 * 210, 2000 + 158 * 30 - 500], "R": 800});
else:
    lims_x = [-np.Inf, np.Inf];
    lims_y = [-np.Inf, np.Inf];

plt.rcParams['axes.facecolor'] = 'w';

# *********************************************
# Step 1: Clustering
if (computeClustering):
    print('Clustering using DBSCAN ...');
    t1 = time.time();

    if (filename[-4:] == 'hdf5'):
        print(basefolder + "/" + filename)
        print()
        f = h5py.File(basefolder + "/" + filename, 'r')
        dset = f['locs'];
        XC = np.stack((dset["x"], dset["y"])).T
    else:
        XC = np.loadtxt(basefolder + filename);

    XC = np.unique(XC, axis=0);
    XC = datascale * XC;
    np.savetxt(results_folder + '/XC_analysed.txt', XC, fmt="%f %f");

    # *********************************************

    if (case == 'protein_ttx_1hr_2'):
        sigma = 0.9;
    else:
        sigma = 0.7;

    DB = DBSCAN(eps=sigma, min_samples=threshold).fit(XC);
    labels_DBSCAN = DB.labels_;
    print('Compute time DBSCAN: ' + str(time.time() - t1) + ' seconds');
    np.savetxt(results_folder + "/" + case + '_labels_DBSCAN.txt', labels_DBSCAN, fmt="%.0f",
               header='DBSCAN time: ' + str(time.time() - t1) + ' seconds');

    print('Clustering using CAML_07VEJJ ...');
    t1 = time.time();
    labels_CAML_07VEJJ = Clustering_CAML('CAML_07VEJJ', XC, datafolder=basefolder);
    print('Compute time CAML_07VEJJ: ' + str(time.time() - t1) + ' seconds');
    np.savetxt(results_folder + "/" + case + '_labels_CAML_07VEJJ.txt', labels_CAML_07VEJJ, fmt="%.0f",
               header='CAML_07VEJJ time: ' + str(time.time() - t1) + ' seconds');

    print('Clustering using CAML_87B144 ...');
    t1 = time.time();
    labels_CAML_87B144 = Clustering_CAML('CAML_87B144', XC, datafolder=basefolder);
    print('Compute time CAML_87B144: ' + str(time.time() - t1) + ' seconds');
    np.savetxt(results_folder + "/" + case + '_labels_CAML_87B144.txt', labels_CAML_87B144, fmt="%.0f",
               header='CAML_87B144 time: ' + str(time.time() - t1) + ' seconds');

    print('Clustering using FINDER ...');

    t1 = time.time();
    threshold = 10;
    dbscanType = 'DbscanLoop';
    FD = Finder(threshold=threshold, algo=dbscanType,
                similarity_score_computation="threshold")  # ,points_per_dimension=20);
    labels_FINDER = FD.fit(XC);
    print('Compute time FINDER: ' + str(time.time() - t1) + ' seconds');
    np.savetxt(results_folder + '/' + case + '_labels_FINDER.txt', labels_FINDER, fmt="%.0f",
               header='FINDER time: ' + str(time.time() - t1) + ' seconds');
    print('Saving the phase space')
    phase_space = FD.phasespace
    phase_space.to_pickle(results_folder + '/' + case + "_phasespace.pkl")

    # ******************************************
    # labels_all = np.asarray([labels_CAML_87B144,labels_DBSCAN,labels_FINDER,labels_CAML_07VEJJ]).T;
    # np.savetxt(results_folder+"/"+case+'_labels.txt',labels_all,fmt="%.0f %.0f %.0f %.0f", header='CAML_87B144 DBSCAN FINDER CAML_07VEJJ');

    sigma_selected_FINDER = FD.selected_parameters['sigma']
    threshold_selected_FINDER = FD.selected_parameters['threshold']

    parameters_selected_Finder = [sigma_selected_FINDER, threshold_selected_FINDER]

    np.savetxt(results_folder + "/" + case + '_FINDER_optimalSigma.txt', parameters_selected_Finder, fmt="%f",
               header='optimal parameters');

    labels_all = np.asarray([labels_CAML_87B144, labels_DBSCAN, labels_FINDER, labels_CAML_07VEJJ]).T;
    np.savetxt(results_folder + "/" + case + '_labels.txt', labels_all, fmt="%.0f %.0f %.0f %.0f",
               header='CAML_87B144 DBSCAN FINDER CAML_07VEJJ');
else:
    XC = np.loadtxt(results_folder + "/" + 'XC_analysed.txt');
    labels_all = np.loadtxt(results_folder + "/" + case + '_labels.txt');
    labels_CAML_87B144 = labels_all[:, 0];
    labels_DBSCAN = labels_all[:, 1];
    labels_FINDER = np.loadtxt(results_folder + "/" + "protein_ttx_1hr_2_labels_FINDER.txt")  # labels_all[:,2];
    labels_CAML_07VEJJ = labels_all[:, 3];

    params_FINDER = np.loadtxt(results_folder + "/" + case + '_FINDER_optimalSigma.txt')
    sigma_selected_FINDER = params_FINDER[0]
    threshold_selected_FINDER = params_FINDER[1]
    print(results_folder + "/" + case + '_FINDER_optimalSigma.txt')
    print(os.path.isfile(results_folder + "/" + case + '_FINDER_optimalSigma.txt'))
    print(sigma_selected_FINDER)
    #sigma_selected_FINDER = float(sigma_selected_FINDER);

markPlot = (XC[:, 0] > lims_x[0]) * (XC[:, 0] < lims_x[1]) * (XC[:, 1] > lims_y[0]) * (XC[:, 1] < lims_y[1]);
print('Loaded ' + str(len(XC)) + ' unique localizations.')

# *********************************************
# Step 3a: Plot
if (plotFullImage):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5));
    ax.scatter(XC[:, 0], XC[:, 1], s=0.02, c='grey', alpha=0.1);
    #    ax.scatter(x=XCplot[mark,0],y=XCplot[mark,1],s=.04,c='grey',alpha=0.1);

    dx = 10000;
    str_ = '$10\mu m$';
    x_0 = np.min(XC[:, 0]) + 5000;
    y_0 = np.min(XC[:, 1]) - 2500;
    ax.plot([x_0, x_0 + dx], [y_0, y_0], 'k')
    ax.annotate(str_, (x_0 + dx / 2, y_0 + 1000), ha='center');  # fontsize='normal

    ax.plot([lims_x[0], lims_x[1]], [lims_y[0], lims_y[0]], 'r');
    ax.plot([lims_x[1], lims_x[1]], [lims_y[0], lims_y[1]], 'r');
    ax.plot([lims_x[1], lims_x[0]], [lims_y[1], lims_y[1]], 'r');
    ax.plot([lims_x[0], lims_x[0]], [lims_y[1], lims_y[0]], 'r');

    ax.set_aspect(1);
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.axis('off');

    plt.text(0.15, 0.85, 'a', transform=plt.gcf().transFigure);
    plt.savefig(results_folder + "/" + case + "_overview.png", dpi=300, bbox_inches="tight");

if (plotFullImageNoNoise):
    PlotImageNoNoise(XC, labels_FINDER, "FINDER");
    PlotImageNoNoise(XC, labels_CAML_87B144, "CAML_87B144");
    PlotImageNoNoise(XC, labels_CAML_07VEJJ, "CAML_07VEJJ");

if (plot10PtDistance):
    print('Compute 10pt-distances..');
    threshold = 10;
    k = threshold + 1;
    # importing NearestNeighbors class 

    # initialize model
    neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    # train for getting nearest neighbour
    neigh.fit(XC);
    dist_, ind = neigh.kneighbors(XC);

    # We have no use of indices here
    # dist is a 2 dimensional array of shape (10000, 9) in which each row is list of length 9. This row contain distances to all 9 nearest points. But we need distance to only 9th nearest point. So
    nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]

    # CD_sorted = np.sort(dist.squareform(dist.pdist(XC)),axis=1);
    # nPt_distance = CD_sorted[:,threshold];

    sigma_min = np.quantile(nPt_distance, 0.1);
    sigma_max = np.quantile(nPt_distance, 0.9);

    # ****Ripley's K-function**********
    # ==> Use R-estimator instead!!
    # print('Compute Ripleys k-function..');    
    # xy_max = np.max(XC,axis=0);
    # xy_min = np.min(XC,axis=0);   
    # dxy    = xy_max - xy_min;
    # area   = dxy[0]*dxy[1];
    # Kest = RipleysKEstimator(area=area,x_max=xy_max[0], y_max=xy_max[1], x_min=xy_min[0], y_min=xy_min[1]);
    # r = np.linspace(0,200, 100)
    # k_ripley = Kest(data=XC, radii=r, mode='none');
    # h_ripley = np.sqrt(k_ripley/np.pi) - r;

    fig, axs = plt.subplots(1, 2, figsize=(15, 7));

    ax = axs[0];

    # ax.plot(r,h_ripley);
    ax.plot(np.arange(len(nPt_distance)), np.sort(nPt_distance));
    # sns.distplot(nPt_distance,ax=ax);
    plt.grid(True);
    ax.axhline(sigma_min, c='r');
    ax.axhline(sigma_max, c='r');
    ax.axhline(sigma_selected_FINDER, c=my_pal['FINDER_loop'], linestyle='dashed');
    ax.set_ylabel('10-point distance [nm]');
    ax.set_xlabel('Point index (ordered)')

    ax = axs[1];
    sns.distplot(nPt_distance, ax=ax);
    plt.grid(True);
    ax.axvline(sigma_min, c='r');
    ax.axvline(sigma_max, c='r');
    ax.axvline(sigma_selected_FINDER, c=my_pal['FINDER_loop'], linestyle='dashed')
    ax.set_xlabel('10-point distance [nm]');
#    ax.set_xlabel('Point index (ordered)')
# plt.xlim(0,1);


if (False):

    if (case == 'protein_ttx_1hr_2'):
        sigmas = [60, sigma_selected_FINDER, 100];
    elif (case == 'neuron'):
        sigmas = [85, sigma_selected_FINDER, 105];
    else:
        sigmas = 0.7;

    gs_kw = dict(width_ratios=[1, 1, 1], height_ratios=[4, 1], hspace=-0.5, wspace=0.1);
    fig, axs = plt.subplots(2, 3, gridspec_kw=gs_kw, figsize=(15, 14));

    threshold = 10;
    for i, sigma in enumerate(sigmas):
        DBL = DbscanLoop(eps=sigma, min_samples=threshold).fit(XC);
        labels_ = DBL.labels_;

        # DB        = DBSCAN(eps=sigma,min_samples=threshold).fit(XC);
        # labels_   = DB.labels_;
        ax = axs[0, i];
        PlotScatter(labels_, XC, ax=ax, ax_dist=axs[1, i], showScaleBar=(i == 0));
        ax.set_title('DBSCAN (noisefree)\n minPts = 10, r = ' + str(np.round(sigma[1], 2)) + ' nm');
    plt.savefig(results_folder + "/" + case + "_DBSCAN_examples.pdf", bbox_inches="tight");
    print('File saved in : ' + results_folder + "/" + case + "_DBSCAN_examples.pdf")
# *********************************************

if (plotMainAnalysis):
    # Step 3b: Plot
    gs_kw = dict(width_ratios=[1, 1, 1], height_ratios=[3, 1], hspace=0., wspace=0.1);
    fig, axs = plt.subplots(2, 3, gridspec_kw=gs_kw, figsize=(9, 5));
    # fig,axs = plt.subplots(2,3,gridspec_kw=gs_kw,figsize=(18,12));


    PlotScatter(labels_FINDER, XC, markPlot, ax=axs[0, 0], ax_dist=axs[1, 0], col=my_pal['FINDER_loop'],
                showScaleBar=True, highlights=highlights);
    axs[0, 0].set_title('FINDER \n (eps =' + str(np.round(sigma_selected_FINDER, 2)) +'nm' +', minPts = ' + str(np.round(threshold_selected_FINDER, 1)) + ')');

    PlotScatter(labels_CAML_07VEJJ, XC, markPlot, ax=axs[0, 1], ax_dist=axs[1, 1], col=my_pal['CAML_07VEJJ'],
                highlights=highlights);
    axs[0, 1].set_title('CAML (07VEJJ)\n ');
    axs[1, 1].set_ylabel('');
    axs[1, 1].set_yticklabels(["", ""]);

    PlotScatter(labels_CAML_87B144, XC, markPlot, ax=axs[0, 2], ax_dist=axs[1, 2], col=my_pal['CAML_87B144'],
                highlights=highlights);
    axs[0, 2].set_title('CAML (87B144)\n  ');
    axs[1, 2].set_ylabel('');
    axs[1, 2].set_yticklabels(["", ""]);

    plt.text(0.05, 0.9, 'b', transform=plt.gcf().transFigure);
    plt.savefig(results_folder + "/" + case + "_results.pdf", bbox_inches="tight");
    print('File saved in : ' + results_folder + "/" + case + "_results.pdf");
