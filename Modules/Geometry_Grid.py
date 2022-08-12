#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:47:54 2020

@author: andreas
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:52:06 2020

@author: andreas
"""
from Geometry_Base import Geometry_Base;
import numpy as np
import scipy.spatial.distance as dist


class Geometry_Grid(Geometry_Base):

    def __init__(self, basefolder, unitCluster_Library, n_side=2, Delta_ratio=1, noise_ratio=0.):

        super().__init__(basefolder, unitCluster_Library);

        self.geometry_name = 'Geometry_Grid';
        self.parameters = {"Delta_ratio": Delta_ratio,
                           "n_side": n_side}
        self.N_clusters = n_side ** 2;
        self.noise_ratio = noise_ratio;

    def GeneratePoints(self, seed):

        n_side = self.parameters["n_side"];
        Delta_ratio = self.parameters["Delta_ratio"];

        np.random.seed(seed);

        XC_all = np.empty([0, 2]);
        labels_ = np.array((), dtype=np.int);
        current_label = 0;

        number_of_template_clusters = len(self.template_clusters);

        D_ = self.GetTypicalDiameter_of_templateClusters();
        Delta = Delta_ratio * D_;

        indices = np.random.randint(low=0, high=number_of_template_clusters, size=(n_side, n_side));

        # Step 1: Select n**2 clusters randomly and distribute on grid
        for i in np.arange(n_side):
            for j in np.arange(n_side):
                index_ = indices[i, j];
                XC_new = self.template_clusters[index_];
                X_Delta = Delta * np.asarray([i, j]);

                XC_all = np.concatenate((XC_all, XC_new + X_Delta));
                labels_ = np.concatenate((labels_, current_label * np.ones((len(XC_new),), dtype=np.int)));

                current_label = current_label + 1;

                # Step 2: Add noise to grid
        N_Noise = np.int(self.noise_ratio * len(XC_all));
        x_n = np.random.uniform(low=np.min(XC_all[:, 0]), high=np.max(XC_all[:, 0]), size=(N_Noise,));
        y_n = np.random.uniform(low=np.min(XC_all[:, 1]), high=np.max(XC_all[:, 1]), size=(N_Noise,))

        X_noise = np.asarray([x_n, y_n]).T;

        XC_all = np.concatenate((XC_all, X_noise));
        labels_ = np.concatenate((labels_, -1 * np.ones((len(X_noise),), dtype=np.int)));

        self.labels_groundtruth = labels_;
        self.XC = XC_all;
        self.N_Noise = N_Noise;
        self.seed = seed;

    def __generateSamplePoints(self):

        params = self.parameters;

        x_ = np.random.uniform(0, params["x_scale"], params["N_SamplePoints"]);
        y_ = params["y_scale_sigma"] * np.sin(3.14 * x_ / params["x_scale"]) + \
             np.random.normal(loc=0.0, scale=params["y_scale_noise"], size=params["N_SamplePoints"]);
        XC_Sample = (np.array([x_, y_])).T;

        self.XC_Sample = XC_Sample;