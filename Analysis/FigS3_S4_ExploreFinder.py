#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:10:20 2020

@author: andreas
"""

import sys

sys.path.append("Modules/")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import seaborn as sns

from Geometry_Grid import Geometry_Grid
from Finder_1d import Finder_1d
from Definitions import get_datafolder

basefolder = get_datafolder() + "/"

plt.rcParams['axes.facecolor'] = 'w'


# ****************************
# Parameters
# threshold = 10
# points_per_dimension = 15;  # vary for Fig. S19


# ****************************

def get_labels(sigma_idx, threshold_idx, PS):
    sigmas = np.unique(PS["sigma"])
    thresholds = np.unique(PS["threshold"])
    rslt_df = PS.loc[(PS['sigma'] == sigmas[sigma_idx]) &
                     (PS['threshold'] == thresholds[threshold_idx])]

    print("sigma = ", sigmas[sigma_idx], " threshold = ", thresholds[threshold_idx])

    return rslt_df.labels.to_numpy()[0]


def PlotScatter(labels, XC, ax=[], showScaleBar=False, showBorder=False):
    # Get correctly detected:
    correct_detected = np.ones_like(labels);
    if (ax == []):
        fig, ax = plt.figure();
    mark = (labels == -1);
    sns.scatterplot(x=XC[mark, 0], y=XC[mark, 1], color='grey', alpha=0.2, ax=ax);
    mark = (labels >= 0);
    sns.scatterplot(x=XC[mark, 0], y=XC[mark, 1], hue=labels[mark], palette='Set1',
                    size=0.2, style=-1 * correct_detected[mark], legend=False, ax=ax);
    ax.set_aspect('equal');

    x_0 = 0;
    y_0 = np.min(XC[:, 1]) - 80;
    if (showScaleBar):
        ax.plot([x_0, x_0 + 100], [y_0, y_0], 'k')
        ax.annotate('$100nm$', (x_0 + 50, y_0 + 10), fontsize='large', ha='center');
    else:
        ax.plot([x_0, x_0 + 100], [y_0, y_0], 'w')

    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    if ax == []:
        plt.show();


# for dbscanType in ['dbscan','DbscanLoop']:

for dbscanType in ['DbscanLoop', 'dbscan']:
    for name_idx in ["FigS3", "FigS4"]:
        name = str(name_idx) + '_' + dbscanType

        if (name_idx == "FigS4"):
            params = {'n_side': 5,
                      'seed': 1,
                      'Delta_ratio': 1.5,
                      'noise_ratio': 1.,
                      'unit_type': 'Clusters_DNA_1mers'};  # "Clusters_DNA_1mers";#"Clusters_Neuron";

        elif (name_idx == "FigS3"):
            params = {'n_side': 5,
                      'seed': 1,
                      'Delta_ratio': 1.5,
                      'noise_ratio': 1.5,
                      'unit_type': 'Clusters_Neuron'};  # "Clusters_DNA_1mers";#"Clusters_Neuron";

        # ****************************
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        filename_dataframe = "Results_" + date_time + ".txt"
        result_folder_name = "Results_" + date_time

        G = Geometry_Grid(basefolder, params['unit_type'],
                          n_side=params['n_side'],
                          Delta_ratio=params['Delta_ratio'],
                          noise_ratio=params['noise_ratio'])
        G.GeneratePoints(params['seed'])

        # basefolder = basefolder + result_folder_name + "/" # CREATE THE DIRECTORY! THIS WILL NOT WORK

        # Test: What does testset look like?
        G.PlotScatter(basefolder + name + "_Groundtruth.pdf");
        XC = G.XC;
        FD = Finder_1d(algo=dbscanType  # ,threshold=threshold,points_per_dimension=points_per_dimension
                       # , one_two_d="oneD"
                       )  # ,points_per_dimension=20);
        labels = FD.fit(XC)

        # *********************************************

        sigmas = np.unique(FD.phasespace["sigma"])
        thresholds = np.unique(FD.phasespace["threshold"])

        sigma_opt = FD.selected_parameters['sigma']
        threshold_opt = FD.selected_parameters['threshold']

        sigma_opt_idx = np.where(sigmas == sigma_opt)[0][0]
        threshold_opt_idx = np.where(thresholds == threshold_opt)[0][0]

        index_opt = np.where(sigmas == sigma_opt)[0][0]

        # Compute similarity Matrix
        similarity = []
        for i, row in FD.phasespace.iterrows():
            similarity.append(row["similarityScore"])
        sim_matr = np.round(np.flipud(np.array(similarity).reshape(15, -1).T), 2)

        # Compute line of optima (max_list) for it
        max_list = []
        for t, s in enumerate(sim_matr.argmax(1)):
            max_list.append(s)
        max_list = max_list[::-1]

        # PLOTTING

        fig, axs = plt.subplots(1, 7, figsize=(14, 3))
        for i, idx_shift in enumerate([-3, -2, -1, 0, 1, 2, 3]):
            # if (index_opt + idx_shift < 0):
            #     axs[i].axis('off')
            #     continue
            # if (index_opt + idx_shift >= len(thresholds)):
            #     axs[i].axis('off')
            #     continue
            threshold_ = thresholds[threshold_opt_idx + idx_shift]
            sigma_ = sigmas[max_list[threshold_opt_idx + idx_shift]]
            labels_ = FD.ComputeClusters(sigma_, threshold_, XC)
            if (i == 0):
                PlotScatter(labels_, XC, ax=axs[i], showScaleBar=False)
            else:
                PlotScatter(labels_, XC, ax=axs[i], showScaleBar=False)
            if (idx_shift == 0):
                axs[i].set_title('minPts =' + str(threshold_) +
                                 '\n eps = ' + str(np.round(sigma_, 1)),
                                 backgroundcolor='silver')
            else:
                axs[i].set_title('minPts =' + str(threshold_) +
                                 '\n eps = ' + str(np.round(sigma_, 1))
                                 )
        plt.text(0.1, 0.85, 'd', fontsize=14, transform=plt.gcf().transFigure)
        plt.savefig(basefolder + name + "_shifts.pdf", bbox_inches="tight");
        plt.show()

        # ***********************************************

        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        plt.tight_layout(pad=4)
        # ground truth
        ax = axs[0]
        PlotScatter(G.labels_groundtruth, XC, ax=ax, showScaleBar=True)
        # phase space
        ax = axs[1]
        sns.heatmap(sim_matr, xticklabels=np.round(sigmas, 2), yticklabels=np.flipud(np.round(thresholds)),
                    ax=ax, cbar=False, cmap='Reds')
        from matplotlib.patches import Rectangle

        for i, j in enumerate(max_list[::-1]):
            ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=3))
        ax.add_patch(
            Rectangle((sigma_opt_idx, len(thresholds) - 1 - threshold_opt_idx),
                      1, 1, fill=False, edgecolor='green', lw=3))
        # sns.lineplot(ax=ax,data=FD.data,x='sigma',y='similarityScore');
        # sns.scatterplot(ax=ax,data=FD.data,x='sigma',y='similarityScore');
        # ax.axvline(sigma_min,c='r');
        # ax.axvline(sigma_max,c='r');
        # ax.axvline(sigma_opt,c='g');
        # #plt.annotate('Selected value', (sigma_opt,FD.phasespace['similarityScore'][index_opt]))
        # trans = ax.get_xaxis_transform()
        # plt.text(sigma_opt, .5, 'Selected value', transform=trans,rotation=90)
        ax.set_xlabel('r')
        ax.set_ylabel('minPts')

        # Similarity score
        ax = axs[2]
        opt_sim = []
        opt_n_cl = []
        for i, j in enumerate(max_list[::-1]):
            opt_sim.append(sim_matr[i, j])
        opt_sim = np.array(opt_sim)
        ax.plot(thresholds,
                (opt_sim[::-1] - opt_sim.min()) / (opt_sim.max() - opt_sim.min()), 'b'
                )
        ax.plot(thresholds,
                (opt_sim[::-1] - opt_sim.min()) / (opt_sim.max() - opt_sim.min()),
                'ob')
        ax.set_xlabel("minPts")
        ax.set_ylabel("Similarity Score (rescaled)")
        ax.axvline(threshold_opt, c='g')
        ax.axhline(0.5, ls='--', c='gray')

        plt.text(0.06, 0.95, 'a', fontsize=14, transform=plt.gcf().transFigure);
        plt.text(0.35, 0.95, 'b', fontsize=14, transform=plt.gcf().transFigure);
        plt.text(0.65, 0.95, 'c', fontsize=14, transform=plt.gcf().transFigure);

        plt.savefig(basefolder + name + "_selection_process.pdf", bbox_inches="tight")
        plt.show()
