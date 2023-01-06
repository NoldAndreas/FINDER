#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:51:07 2020

@author: andreas
"""

import glob
import os
from abc import abstractmethod

# import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
import seaborn as sns

# Geometry base class


class Geometry_Base:
    @abstractmethod
    def __init__(self, basefolder, unitCluster_Library):

        """

        Parameters
        ----------
        basefolder: str
            The path to the folder containing the data.
        unitCluster_Library
            The cluster library one wants to use.
        """

        # Values that have to be set in child classes:
        self.basefolder = basefolder
        self.geometry_name = []
        self.XC = []
        self.N_clusters = []
        self.N_Noise = []
        self.labels_groundtruth = []
        self.parameters = []
        self.unitCluster_Library = unitCluster_Library
        self.seed = []
        self.basefolder = basefolder
        self.geometry_name = []
        self.XC = []
        self.N_clusters = []
        self.N_Noise = []
        self.labels_groundtruth = []
        self.parameters = []
        self.unitCluster_Library = unitCluster_Library
        self.seed = []

        self.__loadUnitClusters()
        self.__loadUnitClusters()

    # Methods that need to be set in child classes:
    @abstractmethod
    def GeneratePoints(self):
        yield None

    def __loadUnitClusters(self):
        data_template_clusters = []
        folder_ = self.unitCluster_Library

        template_clusters_paths = os.path.join(
            self.basefolder, "TemplateClusters", folder_, "cluster_*.txt"
        )

        filenamesList = glob.glob(template_clusters_paths)
        for fn in filenamesList:
            XS_i = np.loadtxt(fn, comments="#", delimiter=" ", unpack=False)
            data_template_clusters.append(XS_i)
        print(
            "Loaded "
            + str(len(data_template_clusters))
            + " template clusters.."
        )

        # In the input data,
        # 1 unit corresponds to 158nm. We normalize such that 1 unit = 1 nanometer
        datascale = 158
        for i, X_cl in enumerate(data_template_clusters):
            data_template_clusters[i] = datascale * (
                X_cl - np.mean(X_cl, axis=0)
            )

        self.template_clusters = data_template_clusters

    def plotScatter(self, labels=None, filename=None):

        if labels is None:
            labels = self.labels_groundtruth
        XC = self.XC

        fig, ax = plt.subplots()

        mark = labels == -1
        sns.scatterplot(
            x=XC[mark, 0], y=XC[mark, 1], color="grey", s=10, alpha=0.2, ax=ax
        )

        mark = labels >= 0
        sns.scatterplot(
            x=XC[mark, 0],
            y=XC[mark, 1],
            hue=labels[mark],
            palette="deep",
            s=10,
            legend=False,
            ax=ax,
        )
        ax.set_aspect("equal")

        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        if filename is not None:
            plt.savefig(filename)
            print(f"saving picture in {filename}")

        else:
            plt.show()

    def GetTypicalDiameter_of_templateClusters(self):
        D_ = 0
        for cl in self.template_clusters:
            d = np.max(dist.pdist(cl))
            if d > D_:
                D_ = d
        return D_

    def GetTypical_Number_of_points_templateClusters(self):
        Ns = [len(cl) for cl in self.template_clusters]

        return np.mean(np.asarray(Ns))


class Geometry_Free(Geometry_Base):
    def __init__(
        self,
        basefolder,
        unitCluster_Library,
        n_side=2,
        Delta_ratio=1,
        noise_ratio=0.0,
    ):

        """

        Parameters
        ----------
        basefolder: str

        unitCluster_Library: str

        n_side: int
            TODO: not used
        Delta_ratio:

        noise_ratio: float
            amount of noise in the dataset.
        """

        super().__init__(basefolder, unitCluster_Library)

        self.geometry_name = "Geometry_Free"
        self.parameters = {"Delta_ratio": Delta_ratio}
        self.noise_ratio = noise_ratio

    def GeneratePoints(self, seed):
        """

        Distribute the points randomly.

        1. Define limits of box and Consecutively assign positions
        2. Add noise

        Parameters
        ----------
        seed:
            The random seed
        """
        # TODO: why is it coded?
        Delta_ratio = 3  # self.parameters["Delta_ratio"];

        np.random.seed(seed)

        D = (
            self.GetTypicalDiameter_of_templateClusters()
        )  # typical diameter of the data
        N_cl = len(self.template_clusters)  # number of data points in the data

        XC_all = np.empty([0, 2])
        labels_ = np.array((), dtype=np.int)

        # ***********************************
        # Define limits of box
        L_box = np.sqrt(N_cl) * D * Delta_ratio
        # ***********************************
        # Consecutively assign positions
        for idx, cl in enumerate(self.template_clusters):
            # Get point:

            while True:
                XC_new = cl + L_box * np.random.uniform(
                    low=0, high=1, size=(1, 2)
                )

                if len(XC_all) == 0:
                    break
                if np.min(dist.cdist(XC_new, XC_all)) > D:
                    break

            XC_all = np.concatenate((XC_all, XC_new))
            labels_ = np.concatenate(
                (labels_, idx * np.ones((len(XC_new),), dtype=np.int))
            )

        # ***********************************
        # Step 2: Add noise
        N_Noise = np.int(self.noise_ratio * len(XC_all))
        x_n = np.random.uniform(
            low=np.min(XC_all[:, 0]),
            high=np.max(XC_all[:, 0]),
            size=(N_Noise,),
        )
        y_n = np.random.uniform(
            low=np.min(XC_all[:, 1]),
            high=np.max(XC_all[:, 1]),
            size=(N_Noise,),
        )

        X_noise = np.asarray([x_n, y_n]).T

        XC_all = np.concatenate((XC_all, X_noise))
        labels_ = np.concatenate(
            (labels_, -1 * np.ones((len(X_noise),), dtype=np.int))
        )

        # ***********************************

        self.labels_groundtruth = labels_
        self.XC = XC_all

        self.N_Noise = N_Noise
        self.seed = seed

    # def __generateSamplePoints(self):

    #     params = self.parameters;

    #     x_ = np.random.uniform(0,params["x_scale"],params["N_SamplePoints"]);
    #     y_ = params["y_scale_sigma"]*np.sin(3.14*x_/params["x_scale"]) +\
    #             np.random.normal(loc=0.0,scale=params["y_scale_noise"],size=params["N_SamplePoints"]);
    #     XC_Sample = (np.array([x_,y_])).T;

    #     self.XC_Sample  = XC_Sample;


class Geometry_Grid(Geometry_Base):
    def __init__(
        self,
        basefolder,
        unitCluster_Library,
        n_side=2,
        Delta_ratio=1,
        noise_ratio=0.0,
    ):

        super().__init__(basefolder, unitCluster_Library)

        self.geometry_name = "Geometry_Grid"
        self.parameters = {"Delta_ratio": Delta_ratio, "n_side": n_side}
        self.N_clusters = n_side**2
        self.noise_ratio = noise_ratio

    def GeneratePoints(self, seed):

        n_side = self.parameters["n_side"]
        Delta_ratio = self.parameters["Delta_ratio"]

        np.random.seed(seed)

        XC_all = np.empty([0, 2])
        labels_ = np.array((), dtype=np.int)
        current_label = 0

        number_of_template_clusters = len(self.template_clusters)

        D_ = self.GetTypicalDiameter_of_templateClusters()
        Delta = Delta_ratio * D_

        indices = np.random.randint(
            low=0, high=number_of_template_clusters, size=(n_side, n_side)
        )

        # Step 1: Select n**2 clusters randomly and distribute on grid
        for i in np.arange(n_side):
            for j in np.arange(n_side):
                index_ = indices[i, j]
                XC_new = self.template_clusters[index_]
                X_Delta = Delta * np.asarray([i, j])

                XC_all = np.concatenate((XC_all, XC_new + X_Delta))
                labels_ = np.concatenate(
                    (
                        labels_,
                        current_label * np.ones((len(XC_new),), dtype=np.int),
                    )
                )

                current_label = current_label + 1

                # Step 2: Add noise to grid
        N_Noise = np.int(self.noise_ratio * len(XC_all))
        x_n = np.random.uniform(
            low=np.min(XC_all[:, 0]),
            high=np.max(XC_all[:, 0]),
            size=(N_Noise,),
        )
        y_n = np.random.uniform(
            low=np.min(XC_all[:, 1]),
            high=np.max(XC_all[:, 1]),
            size=(N_Noise,),
        )

        X_noise = np.asarray([x_n, y_n]).T

        XC_all = np.concatenate((XC_all, X_noise))
        labels_ = np.concatenate(
            (labels_, -1 * np.ones((len(X_noise),), dtype=np.int))
        )

        self.labels_groundtruth = labels_
        self.XC = XC_all
        self.N_Noise = N_Noise
        self.seed = seed

    def __generateSamplePoints(self):

        params = self.parameters

        x_ = np.random.uniform(0, params["x_scale"], params["N_SamplePoints"])
        y_ = params["y_scale_sigma"] * np.sin(
            3.14 * x_ / params["x_scale"]
        ) + np.random.normal(
            loc=0.0,
            scale=params["y_scale_noise"],
            size=params["N_SamplePoints"],
        )
        XC_Sample = (np.array([x_, y_])).T

        self.XC_Sample = XC_Sample


class Geometry_Path(Geometry_Base):
    def __init__(self, basefolder, unitCluster_Library):
        super().__init__(basefolder, unitCluster_Library)

        self.geometry_name = "Geometry_Path"
        self.parameters = {"y_scale_noise": 1, "y_scale": 2, "x_scale": 20}
        # 'N_SamplePoints":2000}

    def GeneratePoints(self, N_clusters, N_Noise, seed):
        """

        Parameters
        ----------
        N_clusters
        N_Noise
        seed

        Returns
        -------

        """

        np.random.seed(seed)

        XC_cluster_centers = self.__generateSamplePoints(N_clusters)
        XC_all, labels, selected_clusters = self.__positionClusters(
            XC_cluster_centers
        )

        XC_noise = self.__generateSamplePoints(N_Noise)

        XC_all = np.concatenate((XC_all, XC_noise))
        labels = np.concatenate(
            (labels, -np.ones((len(XC_noise),), dtype=np.int))
        )

        self.labels_groundtruth = labels
        self.XC = XC_all
        self.N_clusters = N_clusters
        self.N_Noise = N_Noise
        self.seed = seed

    def __generateSamplePoints(self, N):
        params = self.parameters

        D = self.GetTypicalDiameter_of_templateClusters()

        y_scale = params["y_scale"]
        x_scale = params["x_scale"]
        y_scale_noise = params["y_scale_noise"]

        x_ = np.random.uniform(0, x_scale, N)
        y_ = y_scale * np.sin(2 * 3.14 * x_ / x_scale) + np.random.normal(
            loc=0.0, scale=y_scale_noise, size=N
        )
        XC_Sample = D * (np.array([x_, y_])).T

        return XC_Sample

    def __positionClusters(self, XC_cluster_centers):
        template_clusters = self.template_clusters

        XC_all = np.empty([0, 2])
        labels_ = np.array((), dtype=np.int)

        selected_clusters = []

        for idx_cl, cl_c in enumerate(XC_cluster_centers):
            # Select on cluster randomly
            idx = np.random.randint(len(template_clusters))
            # print(idx);
            selected_clusters.append(idx)

            XC_cluster_to_include = template_clusters[idx]
            XC_cluster_to_include = (
                XC_cluster_to_include
                - np.mean(XC_cluster_to_include, axis=0)
                + cl_c
            )

            XC_all = np.concatenate((XC_all, XC_cluster_to_include))
            labels_ = np.concatenate(
                (
                    labels_,
                    idx_cl
                    * np.ones((len(XC_cluster_to_include),), dtype=np.int),
                )
            )

        return XC_all, labels_, selected_clusters
