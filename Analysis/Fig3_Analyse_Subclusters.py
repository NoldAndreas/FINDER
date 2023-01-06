#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 00:11:55 2020

@author: andreas
"""

import sys

sys.path.append("Modules/")

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Clustering import Clustering
from Definitions import get_datafolder, hue_order

basefolder = get_datafolder()

my_pal = {
    "FINDER_loop": "#701ac0",
    "CAML_07VEJJ": "#eabe8e",
    "CAML_87B144": "#d67d1d",
    "FINDER": "#af6eeb",
    "dbscan": "dimgrey",
    "OPTICS": "lightgrey",
}

dict_algo_names_ = {
    "OPTICS": "OPTICS",
    "dbscan": "DBSCAN",
    "CAML_07VEJJ": "CAML (07VEJJ)",
    "CAML_87B144": "CAML (87B144)",
    "FINDER_loop": "FINDER",
    "FINDER": "FINDER (DBSCAN)",
}

base = basefolder + "/Results_Fig3/"
name_3mers = "Results_3mers"
name_4mers = "Results_4mers"


def PlotHistograms(axs, df, rightCol=False):
    max_show = 10
    for i, algo in enumerate(["FINDER_loop", "CAML_07VEJJ", "CAML_87B144"]):
        ax = axs[i]
        mask2 = df["algos"] == algo
        sns.histplot(
            data=df[mask2],
            x="subcl",
            palette=my_pal,
            ax=axs[i],
            discrete=True,
            shrink=0.8,
            color=my_pal[algo],
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0, 25)
        ax.set_xlim(0, 11)
        # ax.axis('off');
        ax.set_frame_on(False)
        ax.plot([0, 11], [0, 0], "k")
        # ax.axes.get_xaxis().set_visible(True)
        if rightCol:
            ax.annotate(
                dict_algo_names_[algo],
                (10.5, 15),
                color=my_pal[algo],
                weight="bold",
                horizontalalignment="right",
            )

    if rightCol:
        axs[0].annotate(
            "", xy=(4, 25), xytext=(4, 30), arrowprops=dict(arrowstyle="->")
        )
    else:
        axs[0].annotate(
            "", xy=(3, 15), xytext=(3, 20), arrowprops=dict(arrowstyle="->")
        )

    ax = axs[2]
    ax.set_xticks(np.arange(max_show + 1))
    xlabs = [str(i) for i in np.arange(max_show + 1)]
    xlabs[-1] = ">= 10"
    ax.set_xticklabels(xlabs)
    ax.set_xlabel("Detected subclusters")


def PlotScatter(labels, XC, ax=[], filename=[]):
    if labels == []:
        labels = -np.ones((len(XC),))

    # Get correctly detected:
    if ax == []:
        fig, ax = plt.subplots()

    ax.scatter(
        x=XC[:, 0],
        y=XC[:, 1],
        marker="o",
        s=1,
        c="grey",
        alpha=0.1,
        edgecolor=None,
        linewidth=0,
    )
    ax.set_aspect("equal")

    x_0 = 0
    y_0 = np.min(XC[:, 1]) - 150
    ax.plot([x_0, x_0 + 500], [y_0, y_0], "k")
    ax.annotate(
        "$500nm$", (x_0 + 250, y_0 + 20), fontsize="large", ha="center"
    )
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    if ax == []:
        plt.show()

    if not (filename == []):
        plt.savefig(filename)


gs_kw = dict(
    width_ratios=[1, 1], height_ratios=[5, 1, 1, 1], hspace=0.0, wspace=0.1
)
fig, axs = plt.subplots(4, 2, gridspec_kw=gs_kw, figsize=(12, 12))

# ****************************************************************
filename_pickle = base + name_3mers + ".pickle"
all_data = []
infile = open(filename_pickle, "rb")
while 1:
    try:
        all_data.append(pickle.load(infile))
    except (EOFError, pickle.UnpicklingError):
        break
infile.close()

no_pts = all_data[0].Geometry.GetTypical_Number_of_points_templateClusters()
print("Typical number of points per 3mer: " + str(no_pts))

PlotScatter([], all_data[0].Geometry.XC, ax=axs[0, 0])
np.savetxt(base + "3mers.txt", all_data[0].Geometry.XC, "%.5f\t %.5f")
axs[0, 0].set_title("3mers")

# ****************************************************************
filename_pickle = base + name_4mers + ".pickle"
all_data = []
infile = open(filename_pickle, "rb")
while 1:
    try:
        all_data.append(pickle.load(infile))
    except (EOFError, pickle.UnpicklingError):
        break
infile.close()

no_pts = all_data[0].Geometry.GetTypical_Number_of_points_templateClusters()
print("Typical number of points per 4mer: " + str(no_pts))
PlotScatter([], all_data[0].Geometry.XC, ax=axs[0, 1])
axs[0, 1].set_title("4mers")
np.savetxt(base + "4mers.txt", all_data[0].Geometry.XC, "%.5f\t %.5f")

# ****************************************************************
filename = name_3mers + "_subclusters0_0.txt"  # "_corrected.txt";

df_3mers = pd.read_csv(base + filename)
ax = axs[1, 0]
mask = df_3mers["subcl"] > 10
df_3mers.loc[mask, "subcl"] = 10
PlotHistograms(axs[1:, 0], df_3mers)

# ****************************************************************
filename = name_4mers + "_subclusters0_0.txt"

df_4mers = pd.read_csv(base + filename)
mask = df_4mers["subcl"] > 10
df_4mers.loc[mask, "subcl"] = 10

PlotHistograms(axs[1:, 1], df_4mers, rightCol=True)

plt.savefig(base + "Fig3_Subclusters.pdf", bbox_inches="tight")
