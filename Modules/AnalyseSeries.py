#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:42:09 2020

@author: andreas
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from PlotScatter import PlotScatter
from PlotScatter import my_pal
import os

hue_order = ['FINDER_loop', 'FINDER', 'CAML_07VEJJ', 'CAML_87B144', 'dbscan', 'OPTICS'];

dict_ylabel = {'true_positives_ratio': 'True positives (ratio)', \
               'false_positives_ratio': 'False positives (ratio)', \
               'compute_time': 'Computation time [seconds]'};

dict_xlabel = {'noise_ratio': 'Noise vs cluster localizations', \
               'Delta_ratio': 'Relative distance between clusters', \
               'N_clusters': 'Number of clusters'};

dict_algo_names_ = {"OPTICS": "OPTICS",
                    "dbscan": "DBSCAN",
                    "CAML_07VEJJ": "CAML (07VEJJ)",
                    "CAML_87B144": "CAML (87B144)",
                    "FINDER_loop": "FINDER  with DBSCAN (noisefree)",
                    "FINDER": "FINDER  with DBSCAN"
                    };


def AnalyseSeries_PlotCases(filename):
    filename_pickle = filename + ".pickle";
    all_data = []
    infile = open(filename_pickle, "rb")
    while 1:
        try:
            all_data.append(pickle.load(infile))
        except (EOFError, pickle.UnpicklingError):
            break
    infile.close()

    basefolder = os.path.dirname(filename) + '/';
    for data in all_data:
        if (data.Geometry.seed >= 1):
            continue;
        filename_ = filename + '_algo_' + data.algo + '_Ncluster_' + str(data.Geometry.N_clusters) + '_Nnoise_' + str(
            data.Geometry.N_Noise) + '_seed_' + str(data.Geometry.seed) + '.pdf';
        PlotScatter(data.labels, data.Geometry.XC, filename=filename_);


def AnalyseSeries(df, params, filename):
    filename_pickle = filename + ".pickle";
    all_data = []
    infile = open(filename_pickle, "rb")
    while 1:
        try:
            all_data.append(pickle.load(infile))
        except (EOFError, pickle.UnpicklingError):
            break
    infile.close()

    # ********************************************************************
    gs_kw = dict(width_ratios=[1, 1], height_ratios=[5, 5, 1], hspace=0.2);
    fig, axs = plt.subplots(nrows=3, ncols=2, gridspec_kw=gs_kw, figsize=(12, 12));  # gridspec_kw=gs_kw

    gs = axs[-1, -1].get_gridspec()
    # remove the underlying axes
    for ax in axs[-1, :]:
        ax.remove()
    ax_legend = fig.add_subplot(gs[-1, :])

    PlotScatter(all_data[0].Geometry.labels_groundtruth, all_data[0].Geometry.XC, ax=axs[0, 0]);
    PlotScatter(all_data[-1].Geometry.labels_groundtruth, all_data[-1].Geometry.XC, ax=axs[0, 1]);

    # ********************************************************************
    sns.set(style="ticks", rc={"lines.linewidth": 0.7});  # style="ticks",

    x_name = params['var_1_name'];

    ax = axs[1, 0];
    y_name = 'true_positives_ratio';
    sns.boxplot(ax=ax, data=df, x=x_name, y=y_name, hue='algo', hue_order=hue_order, palette=my_pal, dodge=True);
    sns.stripplot(ax=ax, x=x_name, y=y_name, data=df, hue='algo', hue_order=hue_order, size=4, palette=my_pal,
                  linewidth=0.5, dodge=True)

    ax.legend([], [], frameon=False);
    ax.set_ylabel(dict_ylabel[y_name]);
    ax.set_xlabel(dict_xlabel[x_name]);
    ax.set_ylim(-0.05, 1.05);

    ax = axs[1, 1];
    y_name = 'false_positives_ratio';
    sns.boxplot(ax=ax, data=df, x=x_name, y=y_name, hue='algo', hue_order=hue_order, palette=my_pal, dodge=True);
    sns.stripplot(ax=ax, x=x_name, y=y_name, data=df, hue='algo', hue_order=hue_order, size=4, palette=my_pal,
                  linewidth=0.5, dodge=True)
    ax.legend([], [], frameon=False);

    ax.set_ylabel(dict_ylabel[y_name]);
    ax.set_xlabel(dict_xlabel[x_name]);
    ax.set_ylim(-0.025, 0.525);

    lines, labels = axs[1, 1].get_legend_handles_labels();
    ax_legend.set_xticks([]);
    ax_legend.set_yticks([]);
    ax_legend.axis('off');

    n_algos = np.int(len(labels) / 2);
    labels_properNames = [dict_algo_names_[l] for l in labels];
    ax_legend.legend(lines[:n_algos], labels_properNames[:n_algos], loc='center', ncol=3, frameon=False)

    plt.savefig(filename + "_analysis.pdf", bbox_inches="tight")
    print("Figure saved in " + filename + "_analysis.pdf")
