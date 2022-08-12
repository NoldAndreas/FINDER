import numpy as np
import time
import pandas as pd
from sklearn.cluster import DBSCAN
from DbscanLoop import DbscanLoop
from sklearn.neighbors import NearestNeighbors
from ProgressBar import printProgressBar
from AuxiliaryFunctions import GetLineOfOptima, GetClusterDistribution
from SimilarityScore import getSimilarityScore, getSimilarityScoreByThreshold
from SimilarityScore import getClusterSizesAll


class Finder_1d:

    def __init__(self, threshold=10, points_per_dimension=15, algo="Dbscan Loop",
                 minmax_threshold= [5, 21],
                 one_two_d="twoD",
                 relative_change_value=0.1,
                 similarity_score_computation="total"
                 , log_thresholds=False
                 , log_sigmas=True
                 , adaptive_sigma_boundaries=False):

        """
        The model class for the `FINDER` algorithm. It uses the sklearn API.
        It explores the two parameters (`threshold` and `sigma`) of the given `algo` and finds the best combination.
        Note that in the paper (and sometimes in the code) the parameters are
        called 'minPts'(=threshold) and 'epsilon'(=sigma) instead.

        Parameters
        ----------
        threshold: int, default = 10
            The default value for the `threshold` value, also called `minPts`.
        points_per_dimension:
            the number of values for each of the two axis  (`threshold` and `sigma`) of the phase space.
        algo: str, default="DbscanLoop"
            the algorithm to be used. Can be either "dbscan" or "Dbscanloop" todo: fix the capital D
        minmax_threshold: list, default = [3,30],
            The minimum and maximum values for the "threshold" parameter.
        one_two_d: str, default = "twoD"
            A string to define the kind of optimization that one wish to carry.
            Possible values are ["oneD", "oneD_thresholds", "twoD"]
        relative_change_value: float, default = 0.1
            The value used to select the best threshold parameter in the "twoD" setting.
            The threshold value is chosen according to the drop of value in the number of clusters.
            When the number of clusters drops by more than `relative_change_value`-percent, that threshold value
            is selected.
        similarity_score_computation: str, default = 'total'
            Select the way in which the similarity score is computed:

            * `total`: compare each parameter combination with any other.
                The number of comparison will then be (p**2)*(p**2 - 1)/2 = O(p**4),
                where p is `points_per_dimension`.
            * 'threshold: compute the similarity only between parameter-combination sharing the same threshold
                (i.e., line by line).
                The number of comparison will then be p*(p)*(p- 1)/2 = O(p**3),
            Applies to the the "twoD" setting only
        log_thresholds: Bool, default = 'False'
            If `True`, the threshold-values are selected in log scale.
        log_sigmas: Bool, default = 'True'
            If `True`, the sigma-values are selected in log scale.

        """

        self.threshold = np.int(threshold)
        self.no_points_sigma = points_per_dimension
        self.no_points_thresholds = points_per_dimension
        self.algo = algo
        self.minmax_threshold = minmax_threshold
        assert one_two_d in ["oneD", "oneD_thresholds", "twoD"], \
            'Possible values are ["oneD", "oneD_thresholds", "twoD"]'
        self.one_two_d = one_two_d
        self.relative_acceptable_change_of_clusters_inFirst_bin = relative_change_value
        assert similarity_score_computation in ["total", "threshold"], \
            'possible values are ["total", "threshold"]'
        self.similarity_score_computation = similarity_score_computation
        self.log_thresholds = log_thresholds
        self.log_sigmas = log_sigmas
        self.adaptive_sigma_boundaries = adaptive_sigma_boundaries

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
            print("skipping the similarity-score computation")
            data = self.__phaseSpacePostProcess(XC, phasespace, skipSimilarityScore=True)

            selth = self.phasespace.loc[np.argmin(np.abs(self.phasespace['threshold'] - self.threshold)), 'threshold']
            mark = (self.phasespace['threshold'] == selth)
            selected_parameters = self.phasespace.loc[self.phasespace.loc[mark, 'no_clusters'].idxmax()]
            labels = selected_parameters['labels']

        else:
            data = self.__phaseSpacePostProcess(XC, phasespace)
            # Step 3: Get parameterset
            if self.one_two_d == "twoD":
                labels, selected_parameters = self.__get_consensus_clustering(data, XC)
            else:
                print('USING 1D OPTIMIZER')
                labels, selected_parameters = self.__get_consensus_clustering_1d(data, XC)

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

    def GetSimilarity(self, labels_1, labels_2):
        # todo: there is no usage of this function
        sim_ = self.__getSimilarity(labels_1, labels_2)
        if (sim_):
            print("similar clusters")
        else:
            print("not similar clusters")
        return sim_

    def getSimilarityScoreDistribution(self, XC, i):

        # todo: this is not used anywhere

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

    def __getParams_SigmasThresholds(self, XC):

        """
        Compute the parameter range for both `threshold` and `sigma`.

        Parameters
        ----------
        XC: the datapoints

        Returns
        -------
        pd.Dataframe
        """

        if self.adaptive_sigma_boundaries:
            minmax_sigma = self.__determine_sigma_boundaries_adaptive(XC)
        else:
            minmax_sigma = self.__determine_sigma_boundaries(XC)

        if self.log_sigmas:
            sigmas = self.__getLogDistribution(minmax_sigma[0], minmax_sigma[1], self.no_points_sigma)
        else:
            sigmas = np.linspace(minmax_sigma[0], minmax_sigma[1], self.no_points_sigma)

        if self.log_thresholds:
            thresholds = np.linspace(self.minmax_threshold[0], self.minmax_threshold[1], self.no_points_sigma)
            thresholds = np.unique(np.round(thresholds))
        else:
            thresholds = np.arange(self.minmax_threshold[0], self.minmax_threshold[1])

        # thresholds = self.__getLogDistribution(self.minmax_threshold[0],
        # self.minmax_threshold[1],
        # self.no_points_thresholds)

        # #LINEAR
        # #sigmas = np.linspace(minmax_sigma[0], minmax_sigma[1], self.no_points_sigma)
        # #thresholds = np.linspace(self.minmax_threshold[0], self.minmax_threshold[1], self.no_points_sigma)
        #
        # #thresholds = np.unique(np.round(thresholds))
        #
        # thresholds = np.arange(self.minmax_threshold[0], self.minmax_threshold[1])
        # print("done")

        s_all = []
        t_all = []
        for s in sigmas:
            for t in thresholds:
                s_all.append(s)
                t_all.append(t)

        print("Sigmas are:")
        print(sigmas)
        print("Thresholds are:")
        print(thresholds)

        params = pd.DataFrame(data={"sigma": np.asarray(s_all),
                                    "threshold": np.asarray(t_all)})
        return params

    def __getParams_Sigmas(self, XC):

        """
        Compute the parameter range for `sigma` for a fixed `threshold` value.
        The value for `threshold` is 'self.threshold'.

        Parameters
        ----------
        XC: the datapoints

        Returns
        -------
        params:pd.Dataframe
            the values of the parameters that FINDER will explore.
        """

        if self.adaptive_sigma_boundaries:
            minmax_sigma = self.__determine_sigma_boundaries_adaptive(XC)
        else:
            minmax_sigma = self.__determine_sigma_boundaries(XC)

        if self.log_sigmas:
            sigmas = self.__getLogDistribution(minmax_sigma[0], minmax_sigma[1], self.no_points_sigma)
        else:
            sigmas = np.linspace(minmax_sigma[0], minmax_sigma[1], self.no_points_sigma)

        params = pd.DataFrame(data={"sigma": sigmas,
                                    "threshold": self.threshold * np.ones_like(sigmas)})
        return params

    def __getParams_Thresholds(self, XC):

        """
        Compute the parameter range for `threshold` for a fixed `sigma` value.
        The value for `sigma` is computed using a KNN.

        Parameters
        ----------
        XC: the datapoints

        Returns
        -------
        params: pd.Dataframe
            the values of the parameters that FINDER will explore.
        """

        k = 10
        # initialize model
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(XC)
        dist_, ind = neigh.kneighbors(XC)

        # We have no use of indices here
        # dist is a 2 dimensional array of shape (10000, 9) in which each row is list of length 9.
        # This row contain distances to all 9 nearest points. But we need distance to only 9th nearest point. So
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]

        # CD_sorted = np.sort(dist.squareform(dist.pdist(XC)),axis=1);
        sigma = np.quantile(nPt_distance, 0.5)

        if self.log_thresholds:
            thresholds = np.linspace(self.minmax_threshold[0], self.minmax_threshold[1], self.no_points_sigma)
            thresholds = np.unique(np.round(thresholds))
        else:
            thresholds = np.arange(self.minmax_threshold[0], self.minmax_threshold[1])

        params = pd.DataFrame(data={"sigma": sigma * np.ones_like(thresholds),
                                    "threshold": thresholds});
        return params

    def __determine_sigma_boundaries_adaptive(self, XC):
        """

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

    def __determine_sigma_boundaries(self, XC):

        """
        OLDER VERSION, THE NEW ONE IS COMMENTED

        Parameters
        ----------
        XC

        Returns
        -------

        """

        k = self.threshold + 1;
        # initialize model
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(XC);
        dist_, ind = neigh.kneighbors(XC);

        # We have no use of indices here
        # dist is a 2 dimensional array of shape (10000, 9) in which each row is list of length 9. This row contain distances to all 9 nearest points. But we need distance to only 9th nearest point. So
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]

        # CD_sorted = np.sort(dist.squareform(dist.pdist(XC)),axis=1);
        sigma_min = np.quantile(nPt_distance, 0.1);
        sigma_max = np.quantile(nPt_distance, 0.9);

        minmax_sigma = [sigma_min, sigma_max];

        print("Boundaries for sigma    : " + str(minmax_sigma[0]) + " , " + str(minmax_sigma[1]));

        return minmax_sigma

    def __getLogDistribution(self, min_x, max_x, n):

        """
        Return an array with the log-distribution of 'n' points between
        `min_x` and `max_x`.
        """
        min_log = np.log(min_x)
        max_log = np.log(max_x)

        log_vec = np.linspace(min_log, max_log, n)

        vec = np.exp(log_vec)
        vec = np.unique(vec)  # why is this needed?

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
        ps: pd.Dataframe
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

    def __phaseSpacePostProcess(self, XC, PS, skipSimilarityScore: bool = False):

        """
        Postprocess the phasespace, computing the similarities between the clusters.

        **steps**:

        1. Initialize Cluster information
        2. Compute similarity scores (optional)

        Parameters
        ----------
        XC:
            The points to be clustered.
        PS: pd.DataFrame
            The phase-space, i.e., the set of labels for each parameter configuration.
        skipSimilarityScore: bool, default=False
            if 'True', avoid computing the similarities.

        Returns
        -------
        PS: pd.DataFrame
            the phasespace with the postporcessing information.

        """

        print("Postprocessing..")

        n = len(PS)

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

            if self.similarity_score_computation == "total":
                cli_similarityScore, similarityScore = getSimilarityScore(XC, PS, clusterInfo)
            if self.similarity_score_computation == "threshold":
                cli_similarityScore, similarityScore = getSimilarityScoreByThreshold(XC, PS, clusterInfo)

        clusterInfo['similarityScore'] = cli_similarityScore
        PS["similarityScore"] = similarityScore

        self.clusterInfo = clusterInfo

        return PS

    def __get_consensus_clustering_1d(self, PS, XC):
        """
        Find the best clustering parameters, defined as the one with the highest `similarityScore`.

        This is supposed to be used when optimizing only one parameter (`sigma` or `threshold`).

        Parameters
        ----------
        PS: pd.DataFrame
            the phasepace
        XC:
            The points (#todo: not used, remove)
        Returns
        -------
        labels, selected_parameters:
            the labels of the chosen  configuration and
            the parameters of the chosen configuration
        """
        # take the similarity scores
        similarity = np.asarray(PS["similarityScore"])
        # find the maximum
        max_score = np.max(similarity)
        # take its index
        idx = np.argwhere(similarity == max_score)[-1][0]
        # store the sigma
        sigma_selected = PS.loc[idx, 'sigma']
        # store the threshold
        threshold_selected = PS.loc[idx, "threshold"]
        # store the labels
        labels = PS.loc[idx, 'labels']

        # store the selected parameters
        selected_parameters = {"sigma": sigma_selected,
                               "threshold": threshold_selected}

        print("Selected threshold , sigma : " + str(threshold_selected) + " , " + str(sigma_selected))
        return labels, selected_parameters

    #
    # def __get_consensus_clustering2(self, PS, XC):
    #     """
    #     Find the best parameter configuration fot the 2-d setting, i.e., when optimizing both 'sigma' and 'threshold'.
    #
    #     **steps**:
    #
    #     1. Get optimal sigma for each theta
    #     2. Get number of clusters in first bin
    #     3. Get theta for which number of clusters in first bin decayed enough
    #
    #     Parameters
    #     ----------
    #     PS: pd.DataFrame
    #         the phasepace
    #     XC:
    #         The points (#todo: not used, remove)
    #     Returns
    #     -------
    #     labels, selected_parameters:
    #         the labels of the chosen  configuration and
    #         the parameters of the chosen configuration
    #     """
    #
    #     # Step 1: Get optimal sigma for each theta
    #     df_opt_th = GetLineOfOptima(PS, 'threshold', 'similarityScore')
    #
    #     # Step 2: Get number of clusters in first bin
    #     noClusters_threshold = []
    #     for index, row in df_opt_th.iterrows():
    #         labels = PS.loc[int(row['idx']), 'labels']
    #         cldist = GetClusterDistribution(labels)
    #
    #         # todo: fix this, comparing the old one and the new one
    #         # # get how many clusters there are with number of localizations equal threshold:
    #         noClusters_threshold.append(np.sum(cldist == row['threshold']))
    #         #noClusters_threshold.append(len(cldist))
    #
    #     noClusters_threshold = np.asarray(noClusters_threshold)
    #     df_opt_th['noClusters_threshold'] = noClusters_threshold
    #
    #     # Step 3: Get theta for which number of clusters in first bin decayed enough
    #
    #     print("NUMBER OF CLUSTERS")
    #     print(noClusters_threshold)
    #
    #     delta_noClusters_threshold = np.abs(noClusters_threshold[1:]-noClusters_threshold[:-1])/noClusters_threshold[0]
    #
    #     idx_chosen = len(delta_noClusters_threshold)
    #     for index, d in enumerate(delta_noClusters_threshold):
    #         if (d < self.relative_acceptable_change_of_clusters_inFirst_bin):
    #             idx_chosen = index + 1  # +1 because we evaluated the difference, so n-1 values, and we take the higher value
    #             break
    #
    #     d_sel = PS.loc[df_opt_th.loc[idx_chosen, 'idx'], :]
    #     labels = d_sel['labels']
    #     selected_parameters = {"sigma": d_sel['sigma'],
    #                            "threshold": d_sel['threshold']}
    #
    #     print("Selected threshold , sigma : " + str(d_sel['threshold']) + " , " + str(d_sel['sigma']))
    #
    #     return labels, selected_parameters
    #

    # def __get_consensus_clustering3(self, PS, XC):
    #
    #     print("You are using Consensus_Clustering_3")
    #
    #     df_opt_th = GetLineOfOptima(PS, 'threshold', 'similarityScore')
    #     line_of_optima = PS.loc[df_opt_th['idx']]
    #     _,sim_score_opt = getSimilarityScore(XC, line_of_optima, self.clusterInfo)
    #
    #     self.sim_score_opt = sim_score_opt
    #
    #     print("SimilarityScores are:")
    #     print(sim_score_opt)
    #     # OLD WAY
    #     #idx_chosen = np.argmax(sim_score_opt)
    #
    #     similarity_diff = np.abs(sim_score_opt[1:]-sim_score_opt[:-1])/sim_score_opt[0]
    #
    #     idx_chosen = len(similarity_diff)
    #     for index, d in enumerate(similarity_diff):
    #         if (d < self.relative_acceptable_change_of_clusters_inFirst_bin):
    #             idx_chosen = index + 1
    #             break
    #
    #
    #
    #     d_sel = PS.loc[df_opt_th.loc[idx_chosen, 'idx'], :]
    #     labels = d_sel['labels']
    #     selected_parameters = {"sigma": d_sel['sigma'],
    #                            "threshold": d_sel['threshold']}
    #
    #     print("Selected threshold , sigma : " + str(d_sel['threshold']) + " , " + str(d_sel['sigma']))
    #
    #     return labels, selected_parameters

    def __get_consensus_clustering(self, PS, XC, decay=0.5):
        """
        Find the best parameter configuration fot the 2-d setting, i.e., when optimizing both 'sigma' and 'threshold'.

        **steps**:

        1. Get optimal sigma for each theta
        2. Normalize the line of optima
        3. choose the theta for which the similarity score is < decay

        Parameters
        ----------
        PS: pd.DataFrame
            the phasepace
        XC:
            The points (#todo: not used, remove)
        decay:
            The threshold value for selecting theta among the optima. Must be in (0,1).
        Returns
        -------
        labels, selected_parameters:
            the labels of the chosen  configuration and
            the parameters of the chosen configuration
        """

        # 1. Get optimal sigma for each theta
        df_opt_th = GetLineOfOptima(PS, 'threshold', 'similarityScore')

        line_of_optima = df_opt_th['idx']
        line_of_optima_sim = np.array(df_opt_th["similarityScore"])
        self.df_opt_th = df_opt_th
        self.line_of_optima = line_of_optima
        # 2. Normalize the diagonal
        opt_normalized = (line_of_optima_sim - line_of_optima_sim.min()) / (
                line_of_optima_sim.max() - line_of_optima_sim.min())

        # 3. choose the theta for which the similarity score is < decay
        ind = np.where(opt_normalized < decay)[0][0]
        optim = PS.iloc[line_of_optima[ind]]

        labels = optim['labels']
        selected_parameters = {"sigma": optim['sigma'],
                               "threshold": optim['threshold']}

        print("Selected threshold , sigma : " + str(optim['threshold']) + " , " + str(optim['sigma']))
        return labels, selected_parameters


class Finder_1d_old:

    # **************************************************
    # **************************************************
    # Public functions
    # **************************************************
    # **************************************************

    # **************************************************
    # Initialization
    # **************************************************
    def __init__(self, threshold=10, points_per_dimension=15, algo="DbscanLoop", one_two_d="HereJustForCompatibility"):
        self.threshold = np.int(threshold);
        self.no_points_sigma = points_per_dimension;
        self.algo = algo;

    # **************************************************
    # fit
    # **************************************************

    def fit(self, XC):

        # Step 1: Get min max of threshold and sigma
        minmax_sigma = self.__determine_sigma_boundaries(XC);

        # Step 2: Compute phase spaces
        phasespace = self.__phaseSpaceLogDomain(XC, minmax_sigma=minmax_sigma);

        # Step 3: Compute similarity score
        data = self.__phaseSpacePostProcess(XC, phasespace);

        # Step 3: Get parameterset
        labels, selected_parameters = self.__get_consensus_clustering(data, XC);

        # Save data
        self.data = data;
        self.labels = labels;
        self.phasespace = phasespace;
        self.selected_parameters = selected_parameters;

        return labels;

    # **************************************************
    # Find Clusters
    # **************************************************
    def GetSimilarity(self, labels_1, labels_2):
        sim_ = self.__getSimilarity(labels_1, labels_2);
        if (sim_):
            print("similar clusters");
        else:
            print("not similar clusters");
        return sim_;

    # **************************************************
    # **************************************************
    # Private functions
    # **************************************************
    # **************************************************

    def __computeCenters_Radii(self, XC, PS):
        centers = [];
        radii = [];
        for i, ps in enumerate(PS):
            no_clusters = np.max(ps["labels"]) + 1;

            centers_i = np.zeros((no_clusters, 2));
            radii_i = np.zeros((no_clusters,));

            # go through all clusters:
            for icl in np.arange(no_clusters):
                XCm = XC[(ps["labels"] == icl)];

                c = np.median(XCm, axis=0);
                centers_i[icl, :] = c;
                radii_i[icl] = np.max(np.linalg.norm(XCm - c, axis=1));
                # np.linalg.norm(np.max(XCm,axis=0) - np.min(XCm,axis=0));

            centers.append(centers_i);
            radii.append(radii_i);

        return centers, radii

    # **************************************************
    # Check if number of points is too low and adjust
    #  minmax_threshold
    # **************************************************
    def __determine_sigma_boundaries(self, XC):

        k = self.threshold + 1;
        # initialize model
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(XC);
        dist_, ind = neigh.kneighbors(XC);

        # We have no use of indices here
        # dist is a 2 dimensional array of shape (10000, 9) in which each row is list of length 9. This row contain distances to all 9 nearest points. But we need distance to only 9th nearest point. So
        nPt_distance = [dist_[i][k - 1] for i in range(len(dist_))]

        # CD_sorted = np.sort(dist.squareform(dist.pdist(XC)),axis=1);
        sigma_min = np.quantile(nPt_distance, 0.1);
        sigma_max = np.quantile(nPt_distance, 0.9);

        minmax_sigma = [sigma_min, sigma_max];

        print("Boundaries for sigma    : " + str(minmax_sigma[0]) + " , " + str(minmax_sigma[1]));

        return minmax_sigma

    # **************************************************
    # __phaseSpaceLogDomain
    # **************************************************
    def __phaseSpaceLogDomain(self, XC, minmax_sigma):

        print("Computing clustering results within sigma boundaries..");
        sigmas = self.__getLogDistribution(minmax_sigma[0], minmax_sigma[1], self.no_points_sigma);
        ps = self.__phaseSpace(sigmas, XC);

        return ps;

    # **************************************************
    # __getLogDistribution
    # **************************************************
    def __getLogDistribution(self, min_x, max_x, n):
        min_log = np.log(min_x);
        max_log = np.log(max_x);

        log_vec = np.linspace(min_log, max_log, n);

        vec = np.exp(log_vec);
        vec = np.unique(vec);

        return vec;

    # **************************************************
    # __phaseSpace
    # **************************************************
    def __phaseSpace(self, sigmas, XC=[]):

        ps = [];
        for sigma in sigmas:
            t1 = time.time()
            labels_ = self.ComputeClusters(sigma, self.threshold, XC);
            t2 = time.time()
            datapoint = {"labels": labels_, "sigma": sigma,
                         "threshold": self.threshold, "time": t2 - t1};
            ps.append(datapoint)

        return ps

    # **************************************************
    # ComputeClusters
    # **************************************************
    def ComputeClusters(self, sigma, threshold, XC):

        if ((self.algo == "dbscan")):
            DB = DBSCAN(eps=sigma, min_samples=threshold).fit(XC);
            labels_ = DB.labels_;
        elif ((self.algo == "DbscanLoop")):
            DBL = DbscanLoop(eps=sigma, min_samples=threshold).fit(XC);
            labels_ = DBL.labels_;
        else:
            self.__print("ALGORITHM NOT RECOGNIZED !!");
        return labels_

    # **************************************************
    # __phaseSpacePostProcess
    #  -- Computes similarity scores
    # **************************************************
    def __phaseSpacePostProcess(self, XC, PS):

        print("Postprocessing..")

        n = len(PS);

        sigmas = np.zeros(shape=(n,))
        thresholds = np.zeros(shape=(n,), dtype=np.int)
        no_clusters = np.zeros(shape=(n,), dtype=np.int)
        no_locs = np.zeros(shape=(n,), dtype=np.int)
        times = np.zeros(shape=(n,))
        similarityScore = np.zeros(shape=(n,))
        similarityScoreMatrix = np.zeros(shape=(n, n))

        # ***************************************
        # Preprocess: get centers and radii
        # ***************************************
        centers, radii = self.__computeCenters_Radii(XC, PS);

        # ***************************
        # Compute similarity scores
        # ***************************
        ###
        t1 = time.time();
        for i in np.arange(n):
            for j in np.arange(i + 1):
                if (not (i == j)):
                    score = self.__getSimilarityScore(i, j, PS, centers, radii);
                    similarityScoreMatrix[j, i] = score;
                    similarityScoreMatrix[i, j] = score;
                else:
                    similarityScoreMatrix[i, j] = np.max(PS[i]["labels"]) + 1;
        print("Computing similarity scores: " + str(time.time() - t1) + " seconds");

        # ***************************
        # Collect data
        # ***************************
        for i, ps in enumerate(PS):
            thresholds[i] = ps["threshold"];
            sigmas[i] = ps["sigma"];
            labels_ = ps["labels"];
            no_clusters[i] = np.int(np.max(labels_) + 1);
            similarityScore[i] = np.sum(similarityScoreMatrix[i, :]);
            times[i] = ps["time"];

        data = np.asarray([no_clusters, no_locs, sigmas, thresholds, times, similarityScore]);
        cols_ = ["no_clusters", "no_locs", "sigma", "thresholds", "time", "similarityScore"];
        df = pd.DataFrame(data.T, columns=cols_);

        return df

        # **************************************************

    # __findRelevantClusters
    # **************************************************
    def __get_consensus_clustering(self, phasespace, XC):

        similarity = np.asarray(phasespace["similarityScore"]);

        max_score = np.max(similarity);
        mark = (similarity == max_score);

        sigma_selected = (np.asarray(phasespace["sigma"]))[mark];
        threshold_selected = (np.asarray(phasespace["thresholds"]))[mark];

        if (np.sum(mark) > 1):
            sigma_selected = sigma_selected[-1];
            threshold_selected = threshold_selected[-1]
        else:
            sigma_selected = sigma_selected[0];
            threshold_selected = threshold_selected[0]

        labels = self.ComputeClusters(sigma_selected, threshold_selected, XC);
        selected_parameters = {"sigma": sigma_selected,
                               "threshold": threshold_selected};

        print("Selected sigma for threshold = " + str(threshold_selected) + " : " + str(sigma_selected));
        return labels, selected_parameters;

    # **************************************************
    # __getSimilarityScore
    # **************************************************
    def __getSimilarityScore(self, i, j, PS, centers, radii):

        labels_1 = PS[i]["labels"];
        labels_2 = PS[j]["labels"];
        radii_1 = radii[i];
        radii_2 = radii[j];
        centers_1 = centers[i];
        centers_2 = centers[j];
        count = 0;

        # return zero score if no clusters selected or
        # if one cluster is selcted which covers most points
        if ((np.max(labels_1) == -1) or (np.max(labels_2) == -1)):
            return count;
        elif ((np.max(labels_1) == 0) and (np.sum(labels_1 == 0) / len(labels_1) > 0.5)):
            return count;
        elif ((np.max(labels_2) == 0) and (np.sum(labels_2 == 0) / len(labels_2) > 0.5)):
            return count;

        # ******************************
        n1 = np.max(labels_1) + 1;
        n2 = np.max(labels_2) + 1;
        similarityMatrix = np.zeros((n1, n2));
        similarityMatrix[:] = np.nan;
        # ******************************

        for i1 in np.arange(n1):
            for i2 in np.arange(n2):

                if (similarityMatrix[i1, i2] == 0):
                    continue;

                if (self.__checkNoOverlapClusters(centers_1[i1, :], centers_2[i2, :], radii_1[i1], radii_2[i2])):
                    similarityMatrix[i1, i2] = 0;
                    continue;

                similarity = self.__getSimilarityClusters(labels_1, labels_2, i1, i2);

                if (similarity):
                    similarityMatrix[i1, :] = 0;
                    similarityMatrix[:, i2] = 0;
                    similarityMatrix[i1, i2] = 1;
                    break;
                else:
                    similarityMatrix[i1, i2] = 0;

        return np.sum(similarityMatrix);

    def __checkNoOverlapClusters(self, c1, c2, r1, r2):
        return (np.linalg.norm(c2 - c1) > r1 + r2);

    def __getSimilarityClusters(self, labels_1, labels_2, i1, i2):
        no_locs_1 = np.sum(labels_1 == i1);
        no_locs_2 = np.sum(labels_2 == i2);
        no_locs_overlap = np.sum((labels_1 == i1) * (labels_2 == i2));

        if ((no_locs_overlap / no_locs_1 > 0.5) and (no_locs_overlap / no_locs_2 > 0.5)):
            return True;
        else:
            return False;
