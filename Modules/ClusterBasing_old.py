import json
import os.path
import pickle

import numpy as np
from FigX4_Explore2DOptimizer_withReference_Streamlined_Functions import (
    DefineCleanedLabels_GeneralLimit,
    FilterPoints,
    GetDensity,
    GetLineOfOptima,
    GetOverlay,
    LoadPoints,
)
from finder import Finder
from SimilarityScore import (
    getClusterSizesAll,
    getSimilarityScore,
    getSimilarityScore_ij,
)


class ClusterBasing:
    def __init__(self, basefolder, parameterfile="parameters"):

        # Load Parameter file
        # parameterfile = 'MikeData/Analysis_dataWindow_1/dataWindow_1_parameters';
        with open(basefolder + parameterfile + ".json") as f:
            parameters = json.load(f)

        if not ("datascale" in parameters.keys()):
            parameters["datascale"] = 1

        parameters["outputfolder"] = (
            parameters["mainfolder"]
            + "Analysis_"
            + parameters["analysis_name"]
            + "/"
        )
        parameters["save_name"] = (
            parameters["outputfolder"] + parameters["analysis_name"]
        )
        parameterfile = (
            basefolder + parameters["save_name"] + "_parameters.json"
        )

        self.basefolder = basefolder
        self.parameters = parameters
        self.save_name = basefolder + parameters["save_name"]

        # Load Points
        self.__loadPoints(basefolder, parameters)

    def __loadPoints(self, basefolder, parameters):

        if os.path.isfile(
            basefolder + parameters["save_name"] + "_filtered_incell.txt"
        ):
            XC_incell = LoadPoints(
                basefolder + parameters["save_name"] + "_filtered_incell.txt",
                datascale=parameters["datascale"],
            )
            XC_outcell = LoadPoints(
                basefolder + parameters["save_name"] + "_filtered_outcell.txt",
                datascale=parameters["datascale"],
            )
        else:
            XC_incell = LoadPoints(
                basefolder
                + parameters["mainfolder"]
                + parameters["image_filename"]
                + "_incell.txt",
                datascale=parameters["datascale"],
            )
            XC_outcell = LoadPoints(
                basefolder
                + parameters["mainfolder"]
                + parameters["image_filename"]
                + "_outcell.txt",
                datascale=parameters["datascale"],
            )

            XC_incell = FilterPoints(XC_incell, parameters["incell_window"])
            XC_outcell = FilterPoints(XC_outcell, parameters["outcell_window"])

            XC_outcell_overlay = GetOverlay(XC_incell, XC_outcell)

            np.savetxt(
                basefolder + parameters["save_name"] + "_filtered_incell.txt",
                XC_incell,
                fmt="%f\t%f",
            )
            np.savetxt(
                basefolder + parameters["save_name"] + "_filtered_outcell.txt",
                XC_outcell,
                fmt="%f\t%f",
            )

        self.XC_incell = XC_incell
        self.XC_outcell = XC_outcell

    def GetClusterings_InOutCell(self):

        parameters = self.parameters
        filename = (
            self.basefolder
            + parameters["outputfolder"]
            + "results_"
            + parameters["analysis_name"]
        )

        # ******************************************************************************************
        # Load or Compute Clustering within cell
        # ******************************************************************************************
        if os.path.exists(filename + "_incell.pickle"):
            with open(filename + "_incell.pickle", "rb") as fr:
                FD_load = pickle.load(fr)
            FD = FD_load["FD"]

            print(
                "Loaded Clustering results from " + filename + "_incell.pickle"
            )
        else:
            FD = Finder(algo=parameters["algo"])
            labels = FD.fit(XC_incell)

            with open(filename + "_incell.pickle", "wb") as handle:
                pickle.dump(
                    {"FD": FD}, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
            print(
                "Computed and saved Clustering results in "
                + filename
                + "_incell.pickle"
            )

        # ******************************************************************************************
        # Load or Compute Clustering outside cell
        # ******************************************************************************************
        if os.path.exists(filename + "_outcell.pickle"):
            with open(filename + "_outcell.pickle", "rb") as fr:
                FD_load = pickle.load(fr)
            FD_ref = FD_load["FD_ref"]

            print(
                "Loaded Clustering results from "
                + filename
                + "_outcell.pickle"
            )
        else:
            FD_ref = Finder(algo=parameters["algo"])
            labels_ref = FD.fit(XC_incell)

            with open(filename + "_outcell.pickle", "wb") as handle:
                pickle.dump(
                    {"FD": FD_ref}, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
            print(
                "Computed and saved Clustering results in "
                + filename
                + "_outcell.pickle"
            )

        # ******************************************************************************************
        # Assemble data
        # ******************************************************************************************
        phasespace_all = FD.phasespace
        phasespace_all["labels_ref"] = FD_ref.phasespace["labels"]
        phasespace_all["no_clusters_ref"] = FD_ref.phasespace["no_clusters"]
        phasespace_all["time_ref"] = FD_ref.phasespace["time"]
        self.phasespace_all = phasespace_all

        df_clusterSizes = FD.clusterInfo
        # GetClusterSizesAll(FD);
        df_clusterSizes_ref = FD_ref.clusterInfo
        # 'GetClusterSizesAll(FD_ref);

        df_clusterSizes["type"] = "incell"
        df_clusterSizes_ref["type"] = "outcell"
        self.df_clusterSizes_all = df_clusterSizes.append(
            df_clusterSizes_ref, ignore_index=True
        )

    def GetReferenceClustering(self, bestRequiredRate=1.0):

        # *********************************************
        # Get limit and filter by cluster size
        # *********************************************
        phasespace_all_aboveT = DefineCleanedLabels_GeneralLimit(
            self.df_clusterSizes_all,
            self.phasespace_all,
            criterion="clusterSize",
            bestRequiredRate=bestRequiredRate,
        )
        clusterInfo_aboveT = getClusterSizesAll(
            self.XC_incell, phasespace_all_aboveT
        )

        cli_similarityScore, similarityScore = getSimilarityScore(
            self.XC_incell, phasespace_all_aboveT, clusterInfo_aboveT
        )

        phasespace_all_aboveT["similarityScore"] = similarityScore
        clusterInfo_aboveT["similarityScore"] = cli_similarityScore

        self.phasespace_all_aboveT = phasespace_all_aboveT
        self.clusterInfo_aboveT = clusterInfo_aboveT
        self.df_opt_th_aboveT_ncl = GetLineOfOptima(
            self.phasespace_all_aboveT[
                ["sigma", "threshold", "similarityScore", "no_clusters"]
            ],
            "threshold",
            "no_clusters",
        )

    def GetClustering(self, criterion="percent_locsIncluded"):  # 'no_clusters'

        df_opt_th_aboveT_ncl = GetLineOfOptima(
            self.phasespace_all_aboveT[
                [
                    "sigma",
                    "threshold",
                    "similarityScore",
                    "no_clusters",
                    "percent_locsIncluded",
                ]
            ],
            "threshold",
            "no_clusters",
        )
        i_choose = df_opt_th_aboveT_ncl.loc[
            df_opt_th_aboveT_ncl[criterion].argmax(), "idx"
        ]
        # i_choose = np.argmax(self.phasespace_all_aboveT[criterion]);
        # i_check = 56;
        v = []
        for i in np.arange(len(self.phasespace_all_aboveT)):
            s1 = getSimilarityScore_ij(
                i_choose,
                i,
                self.phasespace_all_aboveT,
                self.clusterInfo_aboveT,
            )
            #    print(s1)

            if type(s1) != bool:
                v.append(np.sum(s1[0]))
            else:
                v.append(0)
        #       print(np.sum(s1[0]),np.sum(s1[1]));
        self.phasespace_all_aboveT["similarityScoreChosen"] = v
        print(self.phasespace_all_aboveT.loc[i_choose, :])

        return self.phasespace_all_aboveT.loc[i_choose, "labels"]
