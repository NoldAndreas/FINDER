# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from ProgressBar import printProgressBar


def GetSimilarityScorePerCluster(XC, phasespace, df_clusters=[]):
    if len(df_clusters)==0:
        df_clusters = getClusterSizesAll(XC, phasespace)

    cli_similarityScore, similarityScore = getSimilarityScore(XC, phasespace, df_clusters)

    df_clusters['similarityScore'] = cli_similarityScore
    phasespace['similarityScore'] = similarityScore

    return phasespace, df_clusters  # cli_similarityScore,similarityScore; #phaseSpace,df_clusters;


def getSimilarityScore_ij(i, j, PS, clusterInfo):

    '''
    Compute the similarity score between two clustering results.

    Parameters
    ----------
    i: int
        first (row) index
    j: int
        second (col) index
    PS: pd.DataFrame
        the phase space
    clusterInfo

    Returns
    -------
    [s_i, s_j] : list
        The scores for the i and j component.

    '''
    labels_1, labels_2 = PS.loc[i, "labels"], PS.loc[j, "labels"]

    #    clusterInfo_1 = clusterInfo.loc[clusterInfo['index']==i,:];
    #    clusterInfo_2 = clusterInfo.loc[clusterInfo['index']==j,:];
    centers_1 = clusterInfo.loc[clusterInfo['index'] == i, 'center'].values
    centers_2 = clusterInfo.loc[clusterInfo['index'] == j, 'center'].values

    radii_1 = clusterInfo.loc[clusterInfo['index'] == i, 'radius'].values
    radii_2 = clusterInfo.loc[clusterInfo['index'] == j, 'radius'].values

    # return zero score if no clusters selected or
    # if one cluster is selcted which covers most points
    if ((np.max(labels_1) == -1) or (np.max(labels_2) == -1)):
        return False
    elif ((np.max(labels_1) == 0) and (np.sum(labels_1 == 0) / len(labels_1) > 0.5)):
        return False
    elif ((np.max(labels_2) == 0) and (np.sum(labels_2 == 0) / len(labels_2) > 0.5)):
        return False

    # ******************************
    similarityMatrix = np.zeros((np.max(labels_1) + 1, np.max(labels_2) + 1), dtype=int)
    similarityMatrix[:] = -1
    # ******************************

    for i1 in np.unique(labels_1):
        if (i1 == -1): #eclude the noise cluster
            continue
        for i2 in np.unique(labels_2):
            if (i2 == -1): #eclude  the noise cluster
                continue

            if (similarityMatrix[i1, i2] == 0):
                continue

            if (__No_OverlapClusters_Distance(i1, i2, centers_1, centers_2, radii_1, radii_2)):
                similarityMatrix[i1, i2] = 0
                continue

            if (__OverlapClusters_NumberOfLocs(i1, i2, labels_1, labels_2)):
                similarityMatrix[i1, :] = 0
                similarityMatrix[:, i2] = 0
                similarityMatrix[i1, i2] = 1
                break
            else:
                similarityMatrix[i1, i2] = 0

    similarityMatrix[similarityMatrix == -1] = 0

    s_i = np.sum(similarityMatrix, axis=1)
    s_j = np.sum(similarityMatrix, axis=0)
    return [s_i, s_j]


def getSimilarityScore(XC, PS, clusterInfo):
    '''

    Compute similarity score for the whole phase-space.

    **steps**:
    1. Preprocess: get centers and radii
    2. Compute the similarities
    3. Collect data

    Parameters
    ----------
    XC:
        The data points (#TODO: not used, remove)
    PS:
        The phase space information

    clusterInfo: pd.DataFrame
        The information about the dataframe, as produced by `getClusterSizesAll`.

    Returns
    -------

    '''

    t1 = time.time()

    # ***************************************
    # Preprocess: get centers and radii
    # ***************************************
    cli_index = clusterInfo['index']
    cli_similarityScore = np.zeros([len(cli_index), ], dtype=int)

    n = len(PS)
    similarityScoreMatrix = np.zeros(shape=(n, n))
    similarityScore = np.zeros(shape=(n,))
    ps_index = PS.index

    # centers,radii = __computeCenters_Radii(XC,PS);



    # Compute the similarities
    progress_i = 0
    printProgressBar(progress_i, len(PS), prefix='Postprocessing progress:', suffix='Complete', length=50)
    for i, ps in PS.iterrows():
        mark_i = (cli_index == i)
        for j in PS.index[PS.index <= i]:  # np.arange(i+1):
            mark_j = (cli_index == j)

            if (not (i == j)):

                s_ij = getSimilarityScore_ij(i, j, PS, clusterInfo)

                if ((type(s_ij) == bool) and (s_ij == False)):
                    continue

                score = np.sum(s_ij[0])

                cli_similarityScore[mark_i] += s_ij[0]
                cli_similarityScore[mark_j] += s_ij[1]

                similarityScoreMatrix[
                    ps_index == j, ps_index == i] = score  # /Normalize here?  eg  /np.int(np.max(PS.loc[j,"labels"]) + 1)
                similarityScoreMatrix[
                    ps_index == i, ps_index == j] = score  # /Normalize here?  eg  /np.int(np.max(PS.loc[i,"labels"]) + 1)
            # else:
            # we do not include similarity with self

        progress_i += 1
        printProgressBar(progress_i, len(PS), prefix='Progress:', suffix='Complete', length=50)
    print("Computing similarity scores: " + str(np.round(time.time() - t1, 2)) + " seconds")

    # ***************************
    # Collect data
    # ***************************
    for i, ps in PS.iterrows():
        similarityScore[ps_index == i] = np.sum(similarityScoreMatrix[ps_index == i, :])

    return cli_similarityScore, similarityScore


def getClusterSizesAll(XC, PS):
    """

    Parameters
    ----------
    XC:
        The points
    PS:
        The phase-space

    Returns
    -------
    A pd.DataFrame with the info about the clustering.
    """

    cl_sizes = np.array([], dtype=int)
    thresholds = np.array([], dtype=int)
    sigmas = np.array([])
    labels = np.array([], dtype=int)
    index = np.array([], dtype=int)
    centers = np.zeros((0, 2))
    radii = np.array([])

    for idx, df1_row in PS.iterrows():

        labels_i = df1_row['labels']
        l_ = np.arange(np.max(labels_i) + 1)

        if (l_.shape[0] == 0):
            continue

        labels = np.concatenate((labels, l_))
        cl_sizes = np.concatenate((cl_sizes, np.asarray([np.sum(labels_i == l) for l in l_])))
        thresholds = np.concatenate((thresholds, df1_row['threshold'] * np.ones_like(l_)))
        sigmas = np.concatenate((sigmas, df1_row['sigma'] * np.ones([len(l_), ])))
        index = np.concatenate((index, idx * np.ones_like(l_)))

        centers_new, radii_new = __computeCenters_Radii_rowPS(XC, df1_row)
        centers = np.concatenate((centers, centers_new))
        radii = np.concatenate((radii, radii_new))

    df_clusterSizes = pd.DataFrame()
    df_clusterSizes['labels'] = labels
    df_clusterSizes['clusterSize'] = cl_sizes
    df_clusterSizes['threshold'] = thresholds
    df_clusterSizes['sigma'] = sigmas
    df_clusterSizes['index'] = index
    df_clusterSizes['center'] = list(centers)
    df_clusterSizes['radius'] = radii

    return df_clusterSizes


def __No_OverlapClusters_Distance(i1, i2, centers_1, centers_2, radii_1, radii_2):
    c1, c2 = centers_1[i1], centers_2[i2];
    r1, r2 = radii_1[i1], radii_2[i2];
    return (np.linalg.norm(c2 - c1) > r1 + r2);


def __OverlapClusters_NumberOfLocs(i1, i2, labels_1, labels_2):
    no_locs_1 = np.sum(labels_1 == i1)
    no_locs_2 = np.sum(labels_2 == i2)
    no_locs_overlap = np.sum((labels_1 == i1) * (labels_2 == i2));

    if ((no_locs_overlap / no_locs_1 > 0.5) and (no_locs_overlap / no_locs_2 > 0.5)):
        return True
    else:
        return False


def __computeCenters_Radii_rowPS(XC, ps):
    no_clusters = np.max(ps["labels"]) + 1;

    centers_i = np.zeros((no_clusters, 2));
    radii_i = np.zeros((no_clusters,));

    # go through all clusters:
    for icl in np.unique(ps["labels"]):
        if (icl == -1):
            continue;
        XCm = XC[(ps["labels"] == icl)];

        c = np.median(XCm, axis=0);
        centers_i[icl, :] = c;
        radii_i[icl] = np.max(np.linalg.norm(XCm - c, axis=1));

    return centers_i, radii_i


def __computeCenters_Radii(XC,PS):
    centers = []
    radii = []

    for i, ps in PS.iterrows():

        no_clusters = np.max(ps["labels"]) + 1

        centers_i = np.zeros((no_clusters, 2))
        radii_i = np.zeros((no_clusters,))

        # go through all clusters:
        for icl in np.unique(ps["labels"]):
            if (icl == -1):
                continue
            XCm = XC[(ps["labels"] == icl)]

            c = np.median(XCm, axis=0)
            centers_i[icl, :] = c
            radii_i[icl] = np.max(np.linalg.norm(XCm - c, axis=1))

        centers.append(centers_i)
        radii.append(radii_i)

    return centers, radii

