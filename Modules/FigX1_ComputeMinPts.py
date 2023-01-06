#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
import seaborn as sns
from Clustering_CAML import Clustering_CAML
from DbscanLoop import DbscanLoop
from Definitions import data_folder
from Finder import Finder
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# from astropy.stats import RipleysKEstimator
# from varname import nameof

# ************************************************
# ************************************************
# Parameters
# ************************************************
# ************************************************
if True:
    outputfolder = "temp/"
    filename = "/Users/andreas/Documents/NoiseRecognizer_WorkingVersion/ProteinData_ttx_1hr_2/AHA_2_MMStack_Pos0.ome_locs_render_driftcor_filter_render_pix0.02X6f20.hdf5"

    datascale = 158
    # =81/512*1000
    # Define window to analyse
    xInt = [20000, 25000]
    # 30000
    yInt = [20000, 25000]
    # 30000
elif False:
    outputfolder = (
        "/Users/andreas/Documents/NoiseRecognizer_WorkingVersion/MikeData/"
    )
    filename = outputfolder + "XC_red.txt"
    datascale = 1
    # =81/512*1000
    # Define window to analyse
    xInt = [0, np.Infinity]
    yInt = [0, np.Infinity]

thresholds = [10]
# [5,10,15,20];

# ************************************************
# ************************************************
# Functions
# ************************************************
# ************************************************


def GetClusterSizes(labels):
    cluster_sizes = []
    for i in np.unique(labels):
        if i == -1:
            continue
        cluster_sizes.append(np.sum(labels == i))
    cluster_sizes = np.asarray(cluster_sizes)
    return cluster_sizes


def GetDiameters(labels):
    cluster_diams = []
    diams_by_pt = -1 * np.ones_like(labels)
    for i in np.unique(labels):
        if i == -1:
            continue
        diam = np.max(dist.pdist(XC[labels == i, :]))
        cluster_diams.append(diam)
        diams_by_pt[labels == i] = diam

    cluster_diams = np.asarray(cluster_diams)
    return cluster_diams, diams_by_pt


def PlotScatter(info):
    labels = info["labels"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    mark = labels == -1
    ax.scatter(x=XC[mark, 0], y=XC[mark, 1], s=0.4, c="grey", alpha=0.1)

    mark = labels >= 0
    sns.scatterplot(
        x=XC[mark, 0],
        y=XC[mark, 1],
        hue=labels[mark],
        palette="deep",
        linewidth=0,
        s=2,
        legend=False,
        ax=ax,
    )
    ax.set_aspect("equal")
    plt.savefig(
        outputfolder + "results" + str(info["threshold"]) + ".pdf",
        bbox_inches="tight",
    )


def PlotScatterColorSize(info, size):
    labels = info["labels"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    mark = labels == -1
    ax.scatter(x=XC[mark, 0], y=XC[mark, 1], s=0.4, c="grey", alpha=0.1)

    mark = labels >= 0
    norm = plt.Normalize(0, 200)
    ax = sns.scatterplot(
        x=XC[mark, 0],
        y=XC[mark, 1],
        hue=size[mark],
        hue_norm=norm,
        palette="vlag",
        linewidth=0,
        s=2,
        ax=ax,
    )

    sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(sm)

    ax.set_aspect("equal")
    plt.savefig(
        outputfolder + "results" + str(info["threshold"]) + "_size.pdf",
        bbox_inches="tight",
    )


def GetDensities(info, n):

    labels = info["labels"]
    mark = labels >= 0
    h_all, _, _ = np.histogram2d(XC[:, 0], XC[:, 1], bins=n)
    h_clustered, _, _ = np.histogram2d(XC[mark, 0], XC[mark, 1], bins=n)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.scatter(h_all, h_clustered, s=0.1, alpha=0.5)
    plt.plot([0, 600], [0, 600], c="k")
    plt.savefig(
        outputfolder
        + "results"
        + str(info["threshold"])
        + "_densityClusters_vs_all.pdf",
        bbox_inches="tight",
    )


# ************************************************
# ************************************************


if filename[-3:] == "txt":
    XC = np.loadtxt(filename)
elif filename[-4:] == "hdf5":
    # filename = '/Users/andreas/Documents/NoiseRecognizer_WorkingVersion/ProteinData_ttx_1hr_2/AHA_2_MMStack_Pos0.ome_locs_render_driftcor_filter_render_pix0.02X6f20.hdf5';
    f = h5py.File(filename, "r")
    dset = f["locs"]
    XC = np.stack((dset["x"], dset["y"])).T

XC = np.unique(XC, axis=0)
XC = datascale * XC


mark = (
    (XC[:, 0] > xInt[0])
    * (XC[:, 0] < xInt[1])
    * (XC[:, 1] > yInt[0])
    * (XC[:, 1] < yInt[1])
)
XC = XC[mark, :]


if False:
    labels = Clustering_CAML("CAML_87B144", XC, datafolder=basefolder)
    with open(outputfolder + "temp.pickle", "ab") as handle:
        pickle.dump(
            {"labels": labels, "threshold": "CAML_87B144"},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
elif False:
    labels = Clustering_CAML("CAML_07VEJJ", XC, datafolder=basefolder)
    with open(outputfolder + "temp.pickle", "ab") as handle:
        pickle.dump(
            {"labels": labels, "threshold": "CAML_07VEJJ"},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
elif True:
    all_labels = []
    for threshold in thresholds:
        FD = Finder(threshold=threshold, algo="DbscanLoop")
        labels = FD.fit(XC)

        with open(outputfolder + "temp.pickle", "ab") as handle:
            pickle.dump(
                {"labels": labels, "threshold": threshold},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
else:

    # ************************************************
    # ************************************************
    # Plotting
    # ************************************************
    # ************************************************
    # fig,axs = plt.subplots(5,1,figsize=(10,14));
    figS, axsS = plt.subplots(5, 1, figsize=(10, 14))

    i = 0
    with open(outputfolder + "temp.pickle", "rb") as fr:
        try:
            while True:
                all_info = pickle.load(fr)

                PlotScatter(all_info)

                # edges = np.linspace(0,500,101);
                # cluster_sizes = GetClusterSizes(all_info['labels']);
                # sns.histplot(cluster_sizes,ax=axs[i],kde=True,shrink=0.8,bins=edges);
                # axs[i].set_xlim(np.min(edges),np.max(edges));
                # axs[i].set_title(str(all_info['threshold']));

                cluster_sizes, diams_by_pt = GetDiameters(all_info["labels"])
                edges = np.linspace(0, 5, 101)
                sns.histplot(cluster_sizes, ax=axsS[i], kde=True, shrink=0.8)
                # ,bins=edges);
                # axsS[i].set_xlim(np.min(edges),np.max(edges));
                axsS[i].set_title(str(all_info["threshold"]))

                # PlotScatterColorSize(all_info,diams_by_pt);
                # GetDensities(all_info,20);

                i = i + 1

        except EOFError:
            pass

    # fig.savefig(outputfolder+"Distribution_locs.pdf",bbox_inches="tight");
    figS.savefig(outputfolder + "Distribution_size.pdf", bbox_inches="tight")
