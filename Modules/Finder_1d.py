import numpy as np
import time
import pandas as pd
from sklearn.cluster import DBSCAN
from DbscanLoop import DbscanLoop
from sklearn.neighbors import NearestNeighbors
from ProgressBar import printProgressBar
from AuxiliaryFunctions import GetLineOfOptima, GetClusterDistribution
from SimilarityScore import getSimilarityScore
from SimilarityScore import getClusterSizesAll


# **************************************************
# FINDER
#
# Parameters:
# threshold (default = 10)
# points_per_dimension (default = 15)
# algo (default = DbscanLoop)
#
# **************************************************

class Finder_1d:

    # **************************************************
    # **************************************************
    # Public functions
    # **************************************************
    # **************************************************

    # **************************************************
    # Initialization
    # **************************************************
    def __init__(self, threshold=10, points_per_dimension=15, algo="DbscanLoop", minmax_threshold=[3, 30]):

        """
        Parameters
        ----------
        threshold
        points_per_dimension
        algo
        minmax_threshold
        """


        self.threshold = np.int(threshold)
        self.no_points_sigma = points_per_dimension
        self.no_points_thresholds = points_per_dimension
        self.algo = algo
        #todo: why is this hard-coded?
        self.one_two_d = "twoD"
        self.minmax_threshold = minmax_threshold
        self.relative_acceptable_change_of_clusters_inFirst_bin = 0.1

    # **************************************************
    # fit
    # **************************************************

    def fit(self, XC, XC_params=None, **kwargs):
        """

        Generate the labels using FINDER algorithm.

        **Steps**:
         1. Set Boundaries: Get min max of threshold and sigma
         2. Clustering: Compute phase spaces
         3. Postprocessing: Compute similarity score (can be skipped)

        Parameters
        ----------
        XC:
            Points to be clustered
        XC_params:
            The points used to compute the boundaries. If `Null`, `XC` will be used.
        kwargs

        Returns
        -------

        clustering labels

        """
        print("Analysing " + str(len(XC)) + " points")

        if XC_params is None:
            XC_params = XC

        t_1 = time.time()

        # Step 1: Get min max of threshold and sigma
        if self.one_two_d == "oneD":
            params = self.__getParams_Sigmas(XC_params)
        if self.one_two_d == "oneD_thresholds":
            params = self.__getParams_Thresholds(XC_params)
        elif self.one_two_d == "twoD":
            params = self.__getParams_SigmasThresholds(XC_params)

        t_2 = time.time()

        # Step 2: Compute phase spaces
        phasespace = self.__phaseSpace(XC, params)
        self.phasespace = phasespace
        t_3 = time.time()

        # Step 3: Compute similarity score
        if (("skipSimilarityScore" in kwargs.keys()) and (kwargs['skipSimilarityScore'] == True)):
            data = self.__phaseSpacePostProcess(XC, phasespace, skipSimilarityScore=True)

            selth = self.phasespace.loc[np.argmin(np.abs(self.phasespace['threshold'] - self.threshold)), 'threshold']
            mark = (self.phasespace['threshold'] == selth)
            selected_parameters = self.phasespace.loc[self.phasespace.loc[mark, 'no_clusters'].idxmax()]
            labels = selected_parameters['labels']

        else:
            data = self.__phaseSpacePostProcess(XC, phasespace)

            # Step 3: Get parameterset
            labels, selected_parameters = self.__get_consensus_clustering2(data, XC)

        t_4 = time.time()

        # labels,selected_parameters = self.__get_consensus_clustering2(data,XC);
        print("Comp time Step 1 (set boundaries): " + str(np.round(t_2 - t_1, 2)) + " seconds")
        print("Comp time Step 2 (clustering): " + str(np.round(t_3 - t_2, 2)) + " seconds")
        print("Comp time Step 3 (postprocessing): " + str(np.round(t_4 - t_3, 2)) + " seconds")
        print("Selected parameters: " + str(selected_parameters))

        # Save data
        self.computationTimes = {'Step1': t_2 - t_1, 'Step2': t_3 - t_2, 'Step3': t_4 - t_3}
        self.data = data
        self.labels = labels
        self.selected_parameters = selected_parameters

        return labels

    # **************************************************
    # Find Clusters
    # **************************************************
    def GetSimilarity(self, labels_1, labels_2):
        #todo: there is no usage of this function
        sim_ = self.__getSimilarity(labels_1, labels_2);
        if (sim_):
            print("similar clusters");
        else:
            print("not similar clusters");
        return sim_;


    def __getParams_SigmasThresholds(self, XC):
        minmax_sigma = self.__determine_sigma_boundaries(XC)
        sigmas = self.__getLogDistribution(minmax_sigma[0], minmax_sigma[1],
                                           self.no_points_sigma)

        thresholds = self.__getLogDistribution(self.minmax_threshold[0],
                                               self.minmax_threshold[1],
                                               self.no_points_thresholds)

        thresholds = np.unique(np.round(thresholds))

        s_all = []
        t_all = []
        for s in sigmas:
            for t in thresholds:
                s_all.append(s)
                t_all.append(t)

        params = pd.DataFrame(data={"sigma": np.asarray(s_all),
                                    "threshold": np.asarray(t_all)})
        return params

    def __getParams_Sigmas(self, XC):
        minmax_sigma = self.__determine_sigma_boundaries(XC)
        sigmas = self.__getLogDistribution(minmax_sigma[0], minmax_sigma[1], self.no_points_sigma);
        params = pd.DataFrame(data={"sigma": sigmas,
                                    "threshold": self.threshold * np.ones_like(sigmas)});
        return params

    def __getParams_Thresholds(self, XC):

        k = 10
        # initialize model
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(XC)
        dist_, ind = neigh.kneighbors(XC)

        # We have no use of indices here
        # dist is a 2 dimensional array of shape (10000, 9) in which each row is list of length 9. This row contain distances to all 9 nearest points. But we need distance to only 9th nearest point. So
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]

        # CD_sorted = np.sort(dist.squareform(dist.pdist(XC)),axis=1);
        sigma = np.quantile(nPt_distance, 0.5)

        thresholds = self.__getLogDistribution(self.minmax_threshold[0], self.minmax_threshold[1],
                                               self.no_points_sigma)
        thresholds = np.unique(np.round(thresholds))

        params = pd.DataFrame(data={"sigma": sigma * np.ones_like(thresholds),
                                    "threshold": thresholds});
        return params

    # **************************************************
    # Check if number of points is too low and adjust
    #  minmax_threshold
    # **************************************************
    def __determine_sigma_boundaries(self, XC):
        """
        Check if number of points is too low and adjust
        minmax_threshold

        Parameters
        ----------
        XC

        Returns
        -------

        """
        # TO CHANGE!!!
        # todo: change, but what?
        k = self.minmax_threshold[0] + 1  # k         = self.threshold+1;
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        neigh.fit(XC)
        dist_, ind = neigh.kneighbors(XC)
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]
        sigma_min = np.quantile(nPt_distance, 0.1)

        k = self.minmax_threshold[1] + 1  # k         = self.threshold+1;
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        neigh.fit(XC)
        dist_, ind = neigh.kneighbors(XC);
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]
        sigma_max = np.quantile(nPt_distance, 0.9);

        minmax_sigma = [sigma_min, sigma_max];

        print("Boundaries for sigma    : " + str(minmax_sigma[0]) + " , " + str(minmax_sigma[1]))

        return minmax_sigma


    def __getLogDistribution(self, min_x, max_x, n):
        min_log = np.log(min_x)
        max_log = np.log(max_x)

        log_vec = np.linspace(min_log, max_log, n)

        vec = np.exp(log_vec)
        vec = np.unique(vec) #why is this needed?

        return vec


    def __phaseSpace(self, XC, params):
        '''

        Given a set of point to be clustered,
        compute the cluster for each combination of parameters.
        This returns the phase-space,
        which is the set of labels associated with each parameter combination.
        Also computational times are included.

        Parameters
        ----------
        XC:
            data-points
        params:
            list of parameters to compute the phasespace
        Returns
        -------
        the phase space

        '''

        labels_all = []
        times = []

        t1_all = time.time()

        printProgressBar(0, len(params), prefix='Clustering progress:', suffix='Complete', length=50)

        for index, param in params.iterrows():
            t1 = time.time()
            labels_ = self.ComputeClusters(param['sigma'], param['threshold'], XC)
            t2 = time.time()
            labels_all.append(labels_)
            times.append(t2 - t1)
            printProgressBar(index + 1, len(params), prefix='Progress:', suffix='Complete', length=50)
            # print("Computing time for sigma = "+str(np.round(param['sigma'],2))+" and minPts ="+ str(param['threshold'])+" : " + str(np.round(t2-t1,2)) );

        print("Computing clusters : " + str(np.round(time.time() - t1_all, 2)) + " seconds")
        ps = params
        ps['labels'] = labels_all
        ps['time'] = times

        return ps


    def ComputeClusters(self, sigma, threshold, XC):
        """

        Cluster the point using the selected `self.algo` (DBSCAN or Dbscanloop).
        This is basically a iteration of `self.algo` in a given phase-space point.

        Parameters
        ----------
        sigma:
            radius.
        threshold:
            number of point inside the radius.
        XC:
            data-points.

        Returns
        -------

        The labels

        """
        if ((self.algo == "dbscan")):
            DB = DBSCAN(eps=sigma, min_samples=threshold).fit(XC)
            labels_ = DB.labels_

        elif ((self.algo == "DbscanLoop")):
            DBL = DbscanLoop(eps=sigma, min_samples=threshold).fit(XC)
            labels_ = DBL.labels_

        else:
            self.__print("ALGORITHM NOT RECOGNIZED !!");

        return labels_


    def __phaseSpacePostProcess(self, XC, PS, skipSimilarityScore=False):

        """

        Compares the clustering result in the phase-space.

        **steps**:
        1. Preprocess: Initialize Cluster information
        2. Compute similarity scores
        3.

        Parameters
        ----------
        XC:
            The points to be clustered.
        PS: pd.DataFrame
            The phase-space, i.e., the set of labels for each parameter configuration.
        skipSimilarityScore: Bool, default=False
            if 'True', avoid computing the similarities.

        Returns
        -------
        pd.DataFrame

        """

        print("Postprocessing..")

        n = len(PS);

        no_clusters = np.zeros(shape=(n,), dtype=np.int)
        similarityScore = np.zeros(shape=(n,))
        # no_locs                  = np.zeros(shape=(n,),dtype=np.int)
        times = np.zeros(shape=(n,))

        # similarityScoreMatrix    = np.zeros(shape=(n,n))

        # ***********************************************
        # Preprocess: Initialize Cluster information
        # ***********************************************
        clusterInfo = getClusterSizesAll(XC, PS)
        cli_index = clusterInfo['index']
        cli_similarityScore = np.zeros([len(cli_index), ], dtype=int)

        for i, ps in PS.iterrows():
            no_clusters[i] = np.int(np.max(ps["labels"]) + 1)
        PS["no_clusters"] = no_clusters

        # ***************************
        # Compute similarity scores
        # ***************************
        ###

        if (skipSimilarityScore == True):
            for i, ps in PS.iterrows():
                similarityScore[i] = np.nan
        else:
            # ***************************************
            # Preprocess: get centers and radii
            # ***************************************
            cli_similarityScore, similarityScore = getSimilarityScore(XC, PS, clusterInfo)

        clusterInfo['similarityScore'] = cli_similarityScore
        PS["similarityScore"] = similarityScore

        self.clusterInfo = clusterInfo


        return PS


    def __get_consensus_clustering(self, PS, XC):

        similarity = np.asarray(PS["similarityScore"]);

        max_score = np.max(similarity);

        idx = np.argwhere(similarity == max_score)[-1][0]

        sigma_selected = PS.loc[idx, 'sigma'];
        threshold_selected = PS.loc[idx, "threshold"];

        labels = PS.loc[idx, 'labels'];
        selected_parameters = {"sigma": sigma_selected,
                               "threshold": threshold_selected};

        print("Selected threshold , sigma : " + str(threshold_selected) + " , " + str(sigma_selected));
        return labels, selected_parameters;


    def __get_consensus_clustering2(self, PS, XC):

        '''

        Find the "best" clustering parameters.

        **steps**:
        1. Get optimal sigma for each theta
        2. Get number of clusters in first bin
        3. Get theta for which number of clusters in first bin decayed enough

        Parameters
        ----------
        PS: pd.DataFrame
            the phasepace
        XC:
            The points (#todo: not used, remove)
        Returns
        -------
        labels:
            the labels of the chosen chosen configuration
        selected_parameters:
            the parameters of the chosen configuration
        '''

        # Step 1: Get optimal sigma for each theta
        df_opt_th = GetLineOfOptima(PS, 'threshold', 'similarityScore')

        # Step 2: Get number of clusters in first bin
        noClusters_threshold = []
        for index, row in df_opt_th.iterrows():
            labels = PS.loc[int(row['idx']), 'labels']
            cldist = GetClusterDistribution(labels)

            # get how many clusters there are with number of localizations equal threshold:
            noClusters_threshold.append(np.sum(cldist == row['threshold']))

        noClusters_threshold = np.asarray(noClusters_threshold)
        df_opt_th['noClusters_threshold'] = noClusters_threshold

        # Step 3: Get theta for which number of clusters in first bin decayed enough

        delta_noClusters_threshold = np.abs(noClusters_threshold[1:] - noClusters_threshold[:-1]) / \
                                     noClusters_threshold[0]

        idx_chosen = len(delta_noClusters_threshold)
        for index, d in enumerate(delta_noClusters_threshold):
            if (d < self.relative_acceptable_change_of_clusters_inFirst_bin):
                idx_chosen = index + 1;  # +1 because we evaluated the difference, so n-1 values, and we take the higher value
                break

        d_sel = PS.loc[df_opt_th.loc[idx_chosen, 'idx'], :]
        labels = d_sel['labels']
        selected_parameters = {"sigma": d_sel['sigma'],
                               "threshold": d_sel['threshold']}

        return labels, selected_parameters


    def getSimilarityScoreDistribution(self, XC, i):

        PS = self.phasespace;
        print(PS.loc[i, :]);
        labels_1 = PS.loc[i, "labels"];
        centers, radii = self.__computeCenters_Radii(XC, PS);

        n1 = np.max(labels_1) + 1;
        similarityScores = np.zeros((n1,), dtype=int)

        for j, ps in PS.iterrows():
            labels_2 = ps["labels"];

            n1 = np.max(labels_1) + 1;
            n2 = np.max(labels_2) + 1;

            radii_1 = radii[i];
            radii_2 = radii[j];
            centers_1 = centers[i];
            centers_2 = centers[j];

            for i1 in np.arange(n1):
                for i2 in np.arange(n2):
                    similarityScores[i1] += self.__getSimilarityClusters_withPrecheck(labels_1, labels_2, i1, i2,
                                                                                      centers_1, centers_2, radii_1,
                                                                                      radii_2);
        return similarityScores;

#