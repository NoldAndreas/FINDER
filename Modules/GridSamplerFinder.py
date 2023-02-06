#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:44:55 2021

@author: andreas
"""
import time

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats
from finder import Finder
from sklearn.cluster import DBSCAN


class GridSamplerFinder:
    def __init__(self, XC, N_goal=10000):

        self.N_goal = N_goal
        self.XC = XC
        self.TOL = 5
        # accuracy that mean of cluster size distribution is required to be in
        self.delta = 0.05
        # acceptable probability that mean of cluster size distribution is mis-estimated by over TOL
        self.max_clusterSize_toAnalyse = 200
        self.minPointsPerGrid = 100
        self.seed = 0
        self.delta_lim = 0.05
        # error for estimation of mean of cluster size

        N = len(XC)
        n = np.int(np.sqrt(N / N_goal))

        x_min = []
        y_min = []
        x_max = []
        y_max = []

        xy_min = np.min(XC, axis=0)
        xy_max = np.max(XC, axis=0)
        dxy = (xy_max - xy_min) / n

        for i0 in np.arange(n):
            for i1 in np.arange(n):

                # Filter
                x_min_i = xy_min[0] + i0 * dxy[0]
                x_max_i = xy_min[0] + (i0 + 1) * dxy[0]

                y_min_i = xy_min[1] + (i1) * dxy[1]
                y_max_i = xy_min[1] + (i1 + 1) * dxy[1]

                x_min.append(x_min_i)
                x_max.append(x_max_i)
                y_min.append(y_min_i)
                y_max.append(y_max_i)

        df = pd.DataFrame()
        df["xmin"] = x_min
        df["xmax"] = x_max
        df["ymin"] = y_min
        df["ymax"] = y_max

        self.df_limits = df

        np.random.seed(self.seed)
        self.perm = np.random.permutation(len(self.df_limits))

    def __getClusterDistribution(self, labels):
        cl_sizes = []
        for c in np.unique(labels):
            if c == -1:
                continue
            cl_sizes.append(np.sum(labels == c))
        return np.asarray(cl_sizes)

    def sampleGrid(self, algo):

        # ****************************************
        # Initialization
        # ****************************************
        XC = self.XC
        perm = self.perm
        # ****************************************
        cls_dist = np.zeros((0,))

        for i_, p in enumerate(perm):
            # for i,d in self.df_limits.iterrows():
            d = self.df_limits.loc[p, :]
            mark = (
                (XC[:, 0] >= d["xmin"])
                & (XC[:, 0] < d["xmax"])
                & (XC[:, 1] >= d["ymin"])
                & (XC[:, 1] < d["ymax"])
            )

            if np.sum(mark) < self.minPointsPerGrid:
                print(
                    "Grid sample contains "
                    + str(np.sum(mark))
                    + " points (< "
                    + str(self.minPointsPerGrid)
                    + " ), therefore skipped"
                )
                continue

            cls_dist_ = self.__computeClustering(XC[mark, :], algo)
            cls_dist_ = cls_dist_[cls_dist_ < self.max_clusterSize_toAnalyse]
            cls_dist = np.concatenate((cls_dist, cls_dist_))

            # Estimate
            n_cls = len(cls_dist)
            s_ = np.sqrt(n_cls / np.var(cls_dist)) * self.TOL
            delta_est = 2 * (1 - scipy.stats.norm.cdf(s_))
            print("Iteration " + str(i_) + " , delta = " + str(delta_est))

            if delta_est < self.delta_lim:
                break

        return cls_dist

    def __computeClustering(self, XC, algo):

        if algo == "DBSCAN":
            eps = 0.1
            min_samples = 10

            DB = DBSCAN(eps=eps, min_samples=min_samples).fit(XC)
            labels = DB.labels_
        elif algo == "FINDER":
            FD = Finder()
            labels = FD.fit(XC)

        return self.__getClusterDistribution(labels)
