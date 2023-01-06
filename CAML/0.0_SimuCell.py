#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates training and validation data as a 'simulated cell synapse' containing
points in various clustered configurations, as specified.

@author: dave
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

proc_wd = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print("Changed working directory to " + proc_wd)

import FuncEtc as fn_etc
import FuncGenerateData as fn_gendata

# ========================================================================
# General size and format options
# ========================================================================

# Assumes the standard that image axes begin at zero. Therefore ImageSize describes the
# extent (range) of the image field. If you want 2D images, set the third dimension to have
# a range of zero.
# 2D: ImageSize = np.array([40000, 40000, 0])
# 3D: ImageSize = np.array([40000, 40000, 1000])
ImageSize = np.array([40000, 40000, 0])

OutFileExt = ".tsv"
OutDelimiter = "\t"

ExportImageFormat = [
    "png"
]  # Format(s) to save images (given as a Python list).
# Format: 'eps', 'png', 'jpg', 'svg'
# Add multiple formats to the list and each image will
# be saved in that format. For example:
# ExportImageFormat = ['eps','png'] saves EPS and PNG.

# ========================================================================
# Clustering Conditions
# ========================================================================

# Clustering parameters resulting in no clusters (e.g. PointsPerCluster = 0) will
# trigger errors and so will be ignored. Non-clustering (CSR) scenarios will be
# generated for each value of PointsPerMicronSquared specified.
AppendNonClustered = False  # Do not generate any non-clustered scenarios to accompany clustered scenarios

# Scenarios describe clusters which are either:
#   - hard-edged with points uniformly distributed inside min-max radius space
#   - soft-edged with a Gaussian distribution of points.
#     For Gaussian-type clusters, the Radii_min values are ignored, i.e. 'solid' clusters only (no rings) and sigma=0.5*max-radius to ensure most points fall within the 'expected' max radius space.
DistPtsAs = "Uniform"  # either 'Uniform' or 'Gaussian'

### Small demonstration dataset:
TotalReplicates = 3  # Each cluster scenario is repeated this many times.
PointsPerCluster = np.array(
    [10, 20]
)  # Each cluster will contain this many points.
Radii_min = np.array(
    [0]
)  # Nearest a point in the cluster can be to the cluster centre. 0 forms solid clusters
Radii_max = np.array(
    [20, 40]
)  # Furthest a pointin the cluster can be from cluster centre. Effectively cluster radius.
PercentPointsInClusters = np.array(
    [0, 75, 50, 25, 10]
)  # Percentage of all points which are clustered.
PointsPerMicronSquared = np.array(
    [100]
)  # Overall density of points within the 'cell'.

## Training Conditions -- wholly CSR was not used to train
# TotalReplicates = 5
# PointsPerCluster = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
# Radii_min = 0
# Radii_max = np.array([100, 75, 50, 40, 25, 15, 10, 5])
# PercentPointsInClusters = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5])
# PointsPerMicronSquared = np.array([5, 10, 50, 100, 300, 500])

## Evaluation Conditions
# TotalReplicates = 10
# PointsPerCluster = np.array([20])
# Radii_min = np.array([0])
# Radii_max = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
# PercentPointsInClusters = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5])
# PointsPerMicronSquared = np.array([100])

## Scenarios without any clusters -- spatially random data at the given point densities
# TotalReplicates = 3
# PointsPerCluster = np.array([0])
# Radii_min = np.array([0])
# Radii_max = np.array([0])
# PercentPointsInClusters = np.array([0])
# PointsPerMicronSquared = np.array([10, 50, 100, 500, 1000])

#### These settings will triggering scenario culling or similar warnings (for code testing purposes!)
# TotalReplicates = 3
# PointsPerCluster = np.array([0, 10,20]) # has a zero value which will not be used
# Radii_min = np.array([0,30]) # has a min radius which will combine with a max radius where min>max
# Radii_max = np.array([20,40,5000]) # has a max radius which is too large (cannot fit non-overlapped clusters this big)
# PercentPointsInClusters = np.array([200, 75, 50, 25, 10]) # has an invalid percentage value
# PointsPerMicronSquared = np.array([100])

# ========================================================================
# Point Fuzziness
# ========================================================================
# Localisation uncertainty and softening of cluster edges can be specified here:
Uncertainty_fwhm = 0  # in nm.
# Point coordinates will be relocated from their origin by Gaussian function with this fwhm.
# 0 = point coordinates are not modified from their original assignment; use with Edge_fuzz_ratio=0
# to preserve hard-edge structures.

Edge_fuzz_ratio = 0  # fraction of cluster size (radius_max - radius_min)
# Hard-edge clusters will be 'softened' by this amount by relocating points.
# 0 = points are not relocated from their original assigment (after any Uncertainty_fwhm relocations)
# n = points can be reassigned up to n x cluster-size from their origin.
# generally this is below 0.5 to preserve the actual cluster size & area as close
# to the intended area, e.g. when comparing accuracy etc.
# 0.2 gives a 'soft' edge, like a phase-transition.


# ========================================================================
# Scenario Filtering - Exclude wild scenarios
# ========================================================================
# If True, will remove scenarios which are outside the prescribed values (shown
# next) for CpM_min/max and DensityRatio_min/max
FilterScenarios = True

# Scenarios with very low or high clusters-per-um2 can either take too long to
# generate, or be impossile to generate (e.g. if asked to generate more clusters
# than the field of view can accomodate without overlapping) or be functionally
# equivalent to non-clustered scenarios.
CpM_min = 1  # Scenarios resulting in fewer than this many clusters per um2 will be removed
CpM_max = 5  # Scenarios resulting in greater than this many clusters per um2 will be removed

# Density ratio is point-density ratio for the area inside clusters to the area
# outside of clusters.
# If the density ratio is ~1 then there isn't much difference in the point-density
# in clusters to that outside, i.e. clusters are very 'weak'.
DensityRatio_min = (
    1.5  # Scenarios resulting in a density ratio < 1.5 will be removed
)
DensityRatio_max = (
    100  # Scenarios resulting in a density ratio > 100 will be removed
)


# ========================================================================
# Non-homogenous clustering options
# ========================================================================
# create a central region with different clustering. Taken after the T cell synapse
# 'central supramolecular activation complex' or cSMAC.
Create_cSMAC = False
# The following only take effect of Create_cSMAC is True
SMAC_DeleteClustersFromCentral = (
    True  # If False then clustered points will be deleted from cSMAC
)
SMAC_DeleteClustersFraction = 0.95  #
SMAC_DeleteNonCluPtsFromCentral = (
    True  # If False then NC points deleted from outside cSMAC
)
SMAC_DeletePointsFraction = 0.85  #

## Not yet implemented:
# MultipleClusterTypes = False
# DoClusteredClusters = False


# ========================================================================
# Cropped Regions
# ========================================================================
DoCentreCroppedRegions = (
    False  # create centre-cropped subregions of each image?
)
CropSize = 3000  # Regions will be this size (square regions), in nm.
ResetOrigin = False  # Transform coordinates to the range (0,CropSize)?


# ========================================================================
#  End of user-editable variables
# ========================================================================

if __name__ == "__main__":

    starting_scenario = 0  # normally zero. Change it to jump ahead to generating a specific scenario from the list of all condition combinations.

    if type(ImageSize) == int:
        ImageSize = np.array([ImageSize, ImageSize, 0])

    # get the output dir
    outputpath = fn_etc.askforinput(
        message="Path to output folder (will be created if it doesn't exist)",
        errormessage="The output folder must be supplied",
        defaultval="",
        isvalid=lambda v: len(v) > 0 and not v.isspace(),
    )

    doGenerateJSON_answer = fn_etc.askforinput(
        message="Do you want to create a JSON file to accompany these data (required for subsequent processing)?",
        errormessage="Type Y or N",
        defaultval="y",
        isvalid=lambda v: v.lower() in ["y", "n", "yes", "no"],
    )

    if doGenerateJSON_answer.lower() in ["y", "yes"]:
        doGenerateJSON = True
    else:
        doGenerateJSON = False

    if doGenerateJSON:
        NearestNeighbour = fn_etc.askforinput(
            message="Begin measuring distances from this nth nearest-neighbour (Closest Neighbour)",
            errormessage="An integer greater than zero is required",
            defaultval="1",
            isvalid=lambda v: v.isdigit() and int(v) > 0,
        )

        FurthestNeighbour = fn_etc.askforinput(
            message="Finish measuring distances with this nth nearest-neighbour (Furthest Neighbour)",
            errormessage="An integer is required and must be greater than "
            + NearestNeighbour
            + " (Nearest Neighbour)",
            defaultval="100",
            isvalid=lambda v: v.isdigit() and int(v) > int(NearestNeighbour),
        )

    proceed_with_processing = fn_etc.askforinput(
        message="When you are ready to proceed press Enter (or X to cancel everything and exit)",
        errormessage="Type P to proceed or X to exit",
        defaultval="P",
        isvalid=lambda v: v.lower() in ["p", "x"],
    )

    if proceed_with_processing.lower() in ["p"]:
        print("Rightyo, off we go...")
    elif proceed_with_processing.lower() in ["x"]:
        print("That's ok. Maybe next time?")
        raise ValueError(
            "No errors, you just decided not to proceed and that's OK! :)"
        )

    # ========================================================================
    # Create clustering scenarios from combinations of parameters
    # ========================================================================

    # We need to exclude any scenarios which would result in non-clustering
    # but only if we have clustered scenarios present.
    # This is because many combinations of the zero-cluster parameter with the other
    # cluster-generating parameters will end up making essentially the same scenario
    # To avoid this redundant data generation we can skip no-cluster scenarios when
    # there are clustered scenarios available.
    # Non-clustered scenarios can be generated in their own separate run as the only
    # parameter that is in effect there is PointsPerMicronSquared (overall points density, PpMS)

    # We must have non-zero point densities to be able to generate any points at all.
    if any(PointsPerMicronSquared) <= 0:
        PointsPerMicronSquared = np.array(
            [PtsPerMS for PtsPerMS in PointsPerMicronSquared if PtsPerMS > 0]
        )
        fn_etc.warn_msg(
            "PointsPerMicronSquared values must be greater than zero. Invalid values are being ignored."
        )
    if (
        np.max(PointsPerMicronSquared) == 0
        or PointsPerMicronSquared.shape[0] == 0
    ):
        raise ValueError(
            "You must give at least one positive value for the point density ('PointsPerMicronSquared').\n\
                         - Check 'PointsPerMicronSquared' for zeros and try again."
        )

    # Remove any PercentPointsInClusters beyond 100
    if any(PercentPointsInClusters > 100):
        PercentPointsInClusters = np.array(
            [PC for PC in PercentPointsInClusters if PC <= 100]
        )
        fn_etc.warn_msg(
            "PercentPointsInClusters values must be less than 100. Invalid values are being ignored."
        )

    gen_nocluster_hits = (
        np.min(
            np.hstack((PointsPerCluster, PercentPointsInClusters, Radii_max))
        )
        <= 0
    )
    gen_clustered_hits = (
        np.max(
            np.hstack((PointsPerCluster, PercentPointsInClusters, Radii_max))
        )
        != 0
    )

    if gen_nocluster_hits and not gen_clustered_hits:
        csr_scenarios = True  # no clusters, only CSR cenarios

    if not gen_nocluster_hits and gen_clustered_hits:
        csr_scenarios = False  # there are no csr scenarios to generate

    if gen_nocluster_hits and gen_clustered_hits:
        # Keep only positive, non-zero values from these cluster parameters
        # Non-clustered scenarios will be automatically generated for each PointsPerMicronSquared
        PointsPerCluster = np.array(
            [PpC for PpC in PointsPerCluster if PpC > 0]
        )
        PercentPointsInClusters = np.array(
            [PCPtsC for PCPtsC in PercentPointsInClusters if PCPtsC > 0]
        )
        Radii_max = np.array([RadMax for RadMax in Radii_max if RadMax > 0])
        csr_scenarios = False

    AllCombinations = np.array(
        np.meshgrid(
            PointsPerMicronSquared,
            PointsPerCluster,
            PercentPointsInClusters,
            Radii_min,
            Radii_max,
        ),
        dtype=float,
    ).T.reshape(-1, 5)

    # we can keep track of where our data are here. We could use a structured or record array but normal arrays are easier to grow
    PtsPerMicronSqd_col = 0
    PtsPerCluster_col = 1
    PercentPtsClustered_col = 2
    RadiusMin_col = 3
    RadiusMax_col = 4
    #    ClusterApportioning_col = 5
    #    ReplicateID_col = 6

    size_before_filtering = AllCombinations.shape[0]
    if csr_scenarios:
        fn_etc.info_msg(
            str(size_before_filtering)
            + " non-clustering scenarios can created from combinations of the supplied cluster parameters."
        )
    else:
        fn_etc.info_msg(
            str(size_before_filtering)
            + " clustering scenarios can created from combinations of all cluster parameters."
        )

    # remove any scenarios where min radius is larger than max radius
    if np.any(
        AllCombinations[:, RadiusMin_col] < AllCombinations[:, RadiusMax_col]
    ):
        AllCombinations = AllCombinations[
            AllCombinations[:, RadiusMin_col]
            < AllCombinations[:, RadiusMax_col],
            :,
        ]
        size_after_filtering = AllCombinations.shape[0]
        if size_after_filtering < size_before_filtering:
            print(
                "\t\t... "
                + str(size_before_filtering - size_after_filtering)
                + " scenarios removed because their maximum radius was less than the minimum radius."
            )
            size_before_filtering = AllCombinations.shape[0]

    # calculate cluster density for each scenario - this method is clunky but avoids generating divide-by-zero RunTimeWarnings
    #    ClustersPerMicronSqd = np.full((AllCombinations.shape[0],1), np.nan)
    ClustersPerMicronSqd = np.zeros((AllCombinations.shape[0], 1))
    for comboIDX, clusteringcombo in enumerate(AllCombinations):
        if (
            clusteringcombo[PercentPtsClustered_col] > 0
            and clusteringcombo[PtsPerCluster_col] > 0
        ):
            ClustersPerMicronSqd[comboIDX] = (
                clusteringcombo[PtsPerMicronSqd_col]
                * (0.01 * clusteringcombo[PercentPtsClustered_col])
            ) / clusteringcombo[PtsPerCluster_col]

    AllCombinations = np.concatenate(
        (AllCombinations, ClustersPerMicronSqd), axis=1
    )  # glue CpMS to the list of conditions
    ClustersPerMicronSqd_col = 5

    # Filter the list of all combinations to exclude those which result in extremes or unrealistic scenarios
    # E.g. extremely dense points etc

    # Calculate density inside clustered areas
    TotalPointsClusteredPerMicronSqd = AllCombinations[
        :, PtsPerMicronSqd_col, None
    ] * (0.01 * AllCombinations[:, PercentPtsClustered_col, None])
    TotalAreaClusteredPerMicronSqd = (
        AllCombinations[:, RadiusMax_col, None]
        * AllCombinations[:, RadiusMax_col, None]
        * np.pi
        * ClustersPerMicronSqd
        * 1e-6
    )
    if np.all(TotalAreaClusteredPerMicronSqd == 0):
        DensityInClusters = np.zeros((AllCombinations.shape[0], 1))
    else:
        DensityInClusters = (
            TotalPointsClusteredPerMicronSqd / TotalAreaClusteredPerMicronSqd
        )
    AllCombinations = np.concatenate(
        (
            AllCombinations,
            TotalPointsClusteredPerMicronSqd,
            TotalAreaClusteredPerMicronSqd,
            DensityInClusters,
        ),
        axis=1,
    )  # glue CpMS to the list of conditions
    TotalPtsClustered_col = 6
    TotalAreaClustered_col = 7
    PtDensityInClusters_col = 8

    # Calculate density outside of clustered areas
    TotalPointsNotClusteredPerMicronSqd = (
        AllCombinations[:, PtsPerMicronSqd_col, None]
        - TotalPointsClusteredPerMicronSqd
    )
    TotalAreaNotClusteredPerMicronSqd = 1 - TotalAreaClusteredPerMicronSqd
    DensityOutSideClusters = (
        TotalPointsNotClusteredPerMicronSqd / TotalAreaNotClusteredPerMicronSqd
    )
    AllCombinations = np.concatenate(
        (
            AllCombinations,
            TotalPointsNotClusteredPerMicronSqd,
            TotalAreaNotClusteredPerMicronSqd,
            DensityOutSideClusters,
        ),
        axis=1,
    )
    TotalPtsNotClustered_col = 9
    TotalAreaNotClustered_col = 10
    PtDensityOutsideClusters_col = 11

    DensityOutSideClusters[
        DensityOutSideClusters == 0
    ] = (
        np.NaN
    )  # convert zero outside-cluster-densities to NaN to avoid divide-by-zero errors (e.g. when there's 100% points clustered)
    if np.all(DensityInClusters == 0):
        DensityRatio = np.zeros((AllCombinations.shape[0], 1))
    else:
        DensityRatio = DensityInClusters / DensityOutSideClusters
    AllCombinations = np.concatenate((AllCombinations, DensityRatio), axis=1)
    DensityRatio_Col = 12

    del (
        ClustersPerMicronSqd,
        TotalPointsClusteredPerMicronSqd,
        TotalAreaClusteredPerMicronSqd,
        DensityInClusters,
    )
    del (
        TotalPointsNotClusteredPerMicronSqd,
        TotalAreaNotClusteredPerMicronSqd,
        DensityOutSideClusters,
    )

    # Only if we have clustered proposals to look at
    if min(PercentPointsInClusters) > 0 and max(PercentPointsInClusters) != 0:

        # Remove scenarios which require illegal amounts of area,
        # e.g. the total clustered area per 1 micron squared can't be more than 1
        AllCombinations = AllCombinations[
            AllCombinations[:, TotalAreaClustered_col] < 1, :
        ]
        size_after_filtering = AllCombinations.shape[0]
        if size_after_filtering < size_before_filtering:
            print(
                "\t\t... "
                + str(size_before_filtering - size_after_filtering)
                + " scenarios removed for describing clusters that would occupy more area than is available."
            )
            size_before_filtering = AllCombinations.shape[0]

    if FilterScenarios and not csr_scenarios:
        # Keep points within the specified range of clusters per micron squared
        AllCombinations = AllCombinations[
            AllCombinations[:, ClustersPerMicronSqd_col] < CpM_max, :
        ]
        AllCombinations = AllCombinations[
            AllCombinations[:, ClustersPerMicronSqd_col] > CpM_min, :
        ]
        size_after_filtering = AllCombinations.shape[0]
        if size_after_filtering < size_before_filtering:
            print(
                "\t\t... "
                + str(size_before_filtering - size_after_filtering)
                + " scenarios removed for failing the cluster density limits ("
                + str(CpM_min)
                + " < CpMS > "
                + str(CpM_max)
                + ")."
            )
            size_before_filtering = AllCombinations.shape[0]

        # Keep points within the specified range of clusters per micron squared
        AllCombinations = AllCombinations[
            AllCombinations[:, DensityRatio_Col] < DensityRatio_max, :
        ]
        AllCombinations = AllCombinations[
            AllCombinations[:, DensityRatio_Col] > DensityRatio_min, :
        ]
        size_after_filtering = AllCombinations.shape[0]
        if size_after_filtering < size_before_filtering:
            print(
                "\t\t... "
                + str(size_before_filtering - size_after_filtering)
                + " scenarios removed for failing the density ratio limits ("
                + str(DensityRatio_min)
                + " < Density In/Out > "
                + str(DensityRatio_max)
                + ")."
            )
            size_before_filtering = AllCombinations.shape[0]

    # Add the no-clustering (CSR) conditions
    if not csr_scenarios and AppendNonClustered:
        ZeroCombinations = np.zeros(
            (PointsPerMicronSquared.shape[0], AllCombinations.shape[1])
        )
        ZeroCombinations[:, PtsPerMicronSqd_col] = PointsPerMicronSquared

        ZeroCombinations[:, TotalPtsNotClustered_col] = ZeroCombinations[
            :, PtsPerMicronSqd_col
        ]
        ZeroCombinations[:, TotalAreaNotClustered_col] = 1
        ZeroCombinations[:, PtDensityOutsideClusters_col] = ZeroCombinations[
            :, PtsPerMicronSqd_col
        ]
        ZeroCombinations[:, DensityRatio_Col] = np.NaN

        AllCombinations = np.concatenate(
            (ZeroCombinations, AllCombinations), axis=0
        )

        size_after_filtering = AllCombinations.shape[0]
        if size_after_filtering > size_before_filtering:
            print(
                "\t\t... "
                + str(size_after_filtering - size_before_filtering)
                + " non-clustering scenarios were added for each given point-density."
            )

    if TotalReplicates < 1 or not type(TotalReplicates) == int:
        TotalReplicates = 1
        print(
            "\t\t... Value for replicates must be a positive integer (you gave "
            + str(TotalReplicates)
            + "). Setting it to 1 replicate per scenario."
        )

    TotalScenarios = AllCombinations.shape[0]
    TotalSimulations = TotalScenarios * TotalReplicates

    print(
        "\t\t... " + str(TotalScenarios) + " viable scenarios passed checks."
    )

    # ========================================================================
    # Generate data table for each scenario
    # ========================================================================

    if TotalScenarios > 0:
        # make the folder for the output data
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)

        outlines_dir = os.path.join(outputpath, "Cell Outlines")
        cropped_dir = os.path.join(outputpath, "Centre Cropped")
        lucky_fig_dir = os.path.join(outputpath, "Previews")

        if not os.path.exists(outlines_dir):
            os.makedirs(outlines_dir)

        if not os.path.exists(cropped_dir):
            os.makedirs(cropped_dir)

        if not os.path.exists(lucky_fig_dir):
            os.makedirs(lucky_fig_dir)

        fn_etc.info_msg(
            "Generating "
            + str(TotalSimulations)
            + " datasets in total: "
            + str(TotalScenarios)
            + " scenarios, "
            + str(TotalReplicates)
            + " repeats per scenario"
        )
    else:
        raise ValueError(
            "Clustering parameters ultimately combined to create no viable clustering scenarios! Please check the clustering and filtering parameters you have given."
        )

    gnd_truth = np.zeros((TotalSimulations, 16))

    if starting_scenario != 0:
        reset_starting_scenario = fn_etc.askforinput(
            message="The starting scenario is set to"
            + str(starting_scenario)
            + ". Is this correct? (Y or N). If NO then processing will begin with the first scenario in the list",
            errormessage="Type Y or N",
            defaultval="y",
            isvalid=lambda v: v.lower() in ["y", "n", "yes", "no"],
        )

        if reset_starting_scenario.lower() in ["n", "no"]:
            starting_scenario = 0
            print(
                "Starting scenario has been reset and processing will begin from the first scenario in the list."
            )
        else:
            print(
                "Keeping the current scenario. The first scenario generated by this session will be number "
                + str(starting_scenario + 1)
                + " in the list."
            )

    if TotalReplicates > 1:
        # decide which repeat from each simulation will have images saved
        lucky_simRepeat = np.random.randint(0, high=TotalReplicates)
        print(
            "\tImages will be saved for repeat number "
            + str(lucky_simRepeat + 1)
            + " in each scenario."
        )
    elif TotalReplicates == 1:
        doImageEachReplicate = fn_etc.askforinput(
            message="Save an image for every simulation? (Y or N)",
            errormessage="Type Y or N",
            defaultval="n",
            isvalid=lambda v: v.lower() in ["y", "n", "yes", "no"],
        )
        if doImageEachReplicate:
            lucky_simRepeat = 0
    else:
        lucky_simRepeat = 0

    for scenario in range(starting_scenario, TotalScenarios):

        ThisScenario = AllCombinations[scenario, :]

        ptsperum2 = ThisScenario[PtsPerMicronSqd_col]
        ptsperclu = int(ThisScenario[PtsPerCluster_col])
        percentptsinclu = ThisScenario[PercentPtsClustered_col]
        rad_min = ThisScenario[RadiusMin_col]
        rad_max = ThisScenario[RadiusMax_col]
        cluperum2 = ThisScenario[ClustersPerMicronSqd_col]

        for simRepeat in range(0, TotalReplicates):

            scenarioUIDint = (TotalReplicates * scenario) + simRepeat
            scenarioUID = str(scenarioUIDint + 1).zfill(
                len(str(TotalSimulations))
            )

            build_str = (
                scenarioUID
                + "_PpMS("
                + str(ptsperum2)
                + ")_PpC("
                + str(ptsperclu)
                + ")_PC("
                + str(percentptsinclu)
                + ")_r("
                + str(rad_min)
                + "-"
                + str(rad_max)
                + ")_CpMS("
                + str(round(cluperum2, 3))
                + ")"
            )

            if simRepeat == 0:
                print(
                    str(scenario + 1)
                    + "/"
                    + str(TotalScenarios)
                    + "\tCreating "
                    + str(TotalReplicates),
                    "× ["
                    + str(ptsperum2)
                    + " PpMS, "
                    + str(ptsperclu)
                    + " PpC, "
                    + str(percentptsinclu)
                    + " %PC, r("
                    + str(rad_min)
                    + "-"
                    + str(rad_max),
                    "), " + str(round(cluperum2, 4)) + " CpMS]",
                )

            save_file_name = build_str + "_cellID(" + str(simRepeat + 1) + ")"

            if simRepeat == lucky_simRepeat:
                cell_bounds, csmac_bounds = fn_gendata.MakeCellOutline(
                    ImageSize, Create_cSMAC, outlines_dir, save_file_name, True
                )
            else:
                cell_bounds, csmac_bounds = fn_gendata.MakeCellOutline(
                    ImageSize,
                    Create_cSMAC,
                    outlines_dir,
                    save_file_name,
                    False,
                )

            # Save outlines as JSON ROIs
            json_fname = os.path.join(
                outlines_dir, save_file_name + "_ROI_1.json"
            )
            with open(json_fname, "w") as file:
                file.write(json.dumps(cell_bounds.tolist()))

            #    if Create_cSMAC:
            #        json_fname = os.path.join(outlines_dir, save_file_name + '_ROI_1_cSMAC.json')
            #        with open(json_fname, 'w') as file:
            #            file.write(json.dumps(cell_bounds.tolist))

            # contract the outer cell shape to distribute clusters here without risking
            # them landing on the edges
            #    cell_multipoint = MultiPoint(cell_bounds)
            cell_polygon = Polygon(cell_bounds)
            cell_boundingbox = cell_polygon.bounds
            try:
                shrunk_cell_polygon = cell_polygon.buffer(-2.1 * rad_max)[
                    0
                ]  # sometimes shrinking creates a multipolygon. We take the first (large) shape.
            except:
                shrunk_cell_polygon = cell_polygon.buffer(-2.1 * rad_max)
            shrunk_cell_boundary = np.array(shrunk_cell_polygon.exterior.xy).T
            shrunk_cell_boundingbox = shrunk_cell_polygon.bounds

            Ch1_TightCluster_Eventlist, _ = fn_gendata.SprinklePoints(
                cluperum2,
                ptsperclu,
                rad_min,
                rad_max,
                ptsperum2,
                percentptsinclu,
                Uncertainty_fwhm,
                Edge_fuzz_ratio,
                cell_bounds,
                shrunk_cell_polygon,
                csmac_bounds,
                ImageSize,
                DistPtsAs,
                Create_cSMAC,
                DepleteClustersFromcSMAC=True,
                DepleteNCPointsFromcSMAC=False,
                DepleteClustersFraction=0.5,
                DepleteNCPointsFraction=0.75,
            )

            # Check for duplicate points -- this should be dealt with already by SprinklePoints!
            _, uniq_idx = np.unique(
                Ch1_TightCluster_Eventlist[:, :2], axis=0, return_index=True
            )
            if uniq_idx.shape[0] < Ch1_TightCluster_Eventlist.shape[0]:
                uniq_idx = np.sort(uniq_idx)
                print(
                    "\t***\t"
                    + str(
                        Ch1_TightCluster_Eventlist.shape[0] - uniq_idx.shape[0]
                    )
                    + " points with duplicate coordinates were removed."
                )
                Ch1_TightCluster_Eventlist = Ch1_TightCluster_Eventlist[
                    uniq_idx, :
                ]

            # add ChannelID column
            Ch1_TightCluster_Eventlist = np.append(
                Ch1_TightCluster_Eventlist,
                np.ones((Ch1_TightCluster_Eventlist.shape[0], 1)),
                axis=1,
            )

            # save the data table
            if ImageSize[2] == 0:
                data_headers = (
                    "x (nm)"
                    + OutDelimiter
                    + "y (nm)"
                    + OutDelimiter
                    + "Cluster UID"
                    + OutDelimiter
                    + "Class Label"
                    + OutDelimiter
                    + "Channel UID"
                )
                Ch1_TightCluster_Eventlist = np.delete(
                    Ch1_TightCluster_Eventlist, 2, 1
                )
                ClusterUID_col = 2
                LabelID_col = 3

            else:
                data_headers = (
                    "x (nm)"
                    + OutDelimiter
                    + "y (nm)"
                    + OutDelimiter
                    + "z (nm)"
                    + OutDelimiter
                    + "Cluster UID"
                    + OutDelimiter
                    + "Class Label"
                    + OutDelimiter
                    + "Channel UID"
                )
                ClusterUID_col = 3
                LabelID_col = 4

            # Todo: save the uncertainty values
            #            if Uncertainty_fwhm > 0:
            #                    data_headers = data_headers + OutDelimiter + 'Uncertainty'

            data_fname = os.path.join(outputpath, save_file_name + OutFileExt)
            np.savetxt(
                data_fname,
                Ch1_TightCluster_Eventlist,
                delimiter=OutDelimiter,
                fmt="%10.3f",
                header=data_headers,
                comments="",
            )

            # save an image of the clustering
            if simRepeat == lucky_simRepeat:

                Idx_Clu = Ch1_TightCluster_Eventlist[:, 2] > 0
                Idx_Nonclu = Ch1_TightCluster_Eventlist[:, 2] == 0

                plt.ioff()

                if any(Idx_Clu):
                    # plot with clusters coloured (if there are any clusters)
                    fig = plt.figure(figsize=(30, 30))
                    ax = fig.add_subplot(111, facecolor="none")
                    plt.scatter(
                        Ch1_TightCluster_Eventlist[Idx_Nonclu, 0],
                        Ch1_TightCluster_Eventlist[Idx_Nonclu, 1],
                        s=1,
                        color="grey",
                        zorder=1,
                    )
                    plt.scatter(
                        Ch1_TightCluster_Eventlist[Idx_Clu, 0],
                        Ch1_TightCluster_Eventlist[Idx_Clu, 1],
                        s=1,
                        c=Ch1_TightCluster_Eventlist[Idx_Clu, ClusterUID_col],
                    )
                    ax.set_xlim(0, ImageSize[0])
                    ax.set_ylim(0, ImageSize[1])
                    ax.set_aspect("equal")
                    plt.tight_layout()

                    for ExportImage in ExportImageFormat:
                        lucky_fig_fname = os.path.join(
                            lucky_fig_dir,
                            save_file_name
                            + " - Highlight Preview."
                            + ExportImage,
                        )
                        plt.savefig(
                            lucky_fig_fname,
                            dpi=300,
                            bbox_inches=0,
                            facecolor="none",
                            edgecolor="none",
                            transparent=True,
                        )
                    plt.close()

                # plot with all points the same colour
                fig = plt.figure(figsize=(30, 30))
                ax = fig.add_subplot(111, facecolor="none")
                plt.scatter(
                    Ch1_TightCluster_Eventlist[:, 0],
                    Ch1_TightCluster_Eventlist[:, 1],
                    s=1,
                    color="black",
                )
                ax.set_xlim(0, ImageSize[0])
                ax.set_ylim(0, ImageSize[1])
                ax.set_aspect("equal")
                plt.tight_layout()

                for ExportImage in ExportImageFormat:
                    lucky_fig_fname = os.path.join(
                        lucky_fig_dir,
                        save_file_name + " - Flat Preview." + ExportImage,
                    )
                    plt.savefig(
                        lucky_fig_fname,
                        dpi=300,
                        bbox_inches=0,
                        facecolor="none",
                        edgecolor="none",
                        transparent=True,
                    )
                plt.close()

                plt.ion()

            #
            # Update the ground-truth table
            #
            TotalPoints = Ch1_TightCluster_Eventlist.shape[0]
            SimuCellAreaMicronsSqd = cell_polygon.area / 1e6
            ActualClusterCount = (
                np.unique(Ch1_TightCluster_Eventlist[:, ClusterUID_col]).shape[
                    0
                ]
                - 1
            )  # can't use max as some clusters could be deleted from within the csmac (if that option is used) or from outside the cell ROI.
            ActualClustersPerMicron = (
                ActualClusterCount / SimuCellAreaMicronsSqd
            )
            TotalClusteredPoints = np.sum(
                Ch1_TightCluster_Eventlist[:, ClusterUID_col] > 0
            )
            ActualPercentClustered = TotalClusteredPoints / TotalPoints * 100
            ActualInCellPointDensity = (
                np.sum(Ch1_TightCluster_Eventlist[:, ClusterUID_col] > -1)
                / SimuCellAreaMicronsSqd
            )

            gnd_truth_tmp = np.zeros((1, 16))

            gnd_truth_tmp[0, 0] = scenarioUID  # simulation UID
            gnd_truth_tmp[0, 1] = ptsperum2  # Points per micron squared
            gnd_truth_tmp[0, 2] = ptsperclu  # Points per cluster
            gnd_truth_tmp[
                0, 3
            ] = percentptsinclu  # Percentage points clustered
            gnd_truth_tmp[0, 4] = rad_min  # radius min (from seed)
            gnd_truth_tmp[0, 5] = rad_max  # radius max (from seed)
            gnd_truth_tmp[0, 6] = cluperum2  # clusters per micron squared
            gnd_truth_tmp[0, 7] = simRepeat  # repeat number for that scenario
            gnd_truth_tmp[
                0, 8
            ] = ActualClusterCount  # Total number of clusters
            gnd_truth_tmp[
                0, 9
            ] = ActualClustersPerMicron  # Actual cluster density in this image (clusters/um2)
            gnd_truth_tmp[
                0, 10
            ] = ActualPercentClustered  # Actual percent-points-clustered in this image
            gnd_truth_tmp[0, 12] = TotalPoints  # Total points in this image
            gnd_truth_tmp[
                0, 13
            ] = TotalClusteredPoints  # Total number of points in clusters (Cluster UID > 0)
            gnd_truth_tmp[
                0, 13
            ] = SimuCellAreaMicronsSqd  # Area of the simulated cell (in um2)
            gnd_truth_tmp[
                0, 14
            ] = ActualInCellPointDensity  # Actual point-density inside the cell-shape (label > -1)
            if ImageSize[2] == 0:
                gnd_truth_tmp[0, 15] = TotalPoints / (
                    (ImageSize[0] * ImageSize[1]) / 1e6
                )  # Overall point density in the image
            else:
                gnd_truth_tmp[0, 15] = TotalPoints / (
                    (ImageSize[0] * ImageSize[1] * ImageSize[2]) / 1e9
                )  # Overall point density in the image

            gnd_truth[scenarioUIDint, :] = gnd_truth_tmp

            #
            # make a centre-cropped RoI for comparison methods
            #
            if DoCentreCroppedRegions:

                centrepoint_x = ImageSize[0] / 2
                centrepoint_y = ImageSize[1] / 2

                min_x = centrepoint_x - (CropSize / 2)
                max_x = centrepoint_x + (CropSize / 2)
                min_y = centrepoint_y - (CropSize / 2)
                max_y = centrepoint_y + (CropSize / 2)

                # crop in X
                xcropMin = Ch1_TightCluster_Eventlist[:, 0] >= min_x
                xcropMax = Ch1_TightCluster_Eventlist[:, 0] <= max_x
                xcrop = np.array([a & b for (a, b) in zip(xcropMin, xcropMax)])

                # crop in Y
                ycropMin = Ch1_TightCluster_Eventlist[:, 1] >= min_y
                ycropMax = Ch1_TightCluster_Eventlist[:, 1] <= max_y
                ycrop = np.array([a & b for (a, b) in zip(ycropMin, ycropMax)])

                xycrop = np.array([a & b for (a, b) in zip(xcrop, ycrop)])

                Cropped_Ch1_TightCluster_Eventlist = (
                    Ch1_TightCluster_Eventlist[xycrop, :]
                )

                if ResetOrigin:
                    Cropped_Ch1_TightCluster_Eventlist[:, 0] = (
                        Cropped_Ch1_TightCluster_Eventlist[:, 0] - min_x
                    )
                    Cropped_Ch1_TightCluster_Eventlist[:, 1] = (
                        Cropped_Ch1_TightCluster_Eventlist[:, 1] - min_y
                    )

                    min_x = 0
                    max_x = CropSize
                    min_y = 0
                    max_y = CropSize

                # save an image of the clustering
                if simRepeat == lucky_simRepeat:
                    Idx_Clu = Cropped_Ch1_TightCluster_Eventlist[:, 2] > 0
                    Idx_Nonclu = Cropped_Ch1_TightCluster_Eventlist[:, 2] == 0

                    plt.ioff()

                    if any(Idx_Clu):
                        # with clusters coloured
                        fig = plt.figure(figsize=(6, 6))
                        ax = fig.add_subplot(111, facecolor="none")
                        plt.scatter(
                            Cropped_Ch1_TightCluster_Eventlist[Idx_Nonclu, 0],
                            Cropped_Ch1_TightCluster_Eventlist[Idx_Nonclu, 1],
                            s=1,
                            color="grey",
                            zorder=1,
                        )
                        plt.scatter(
                            Cropped_Ch1_TightCluster_Eventlist[Idx_Clu, 0],
                            Cropped_Ch1_TightCluster_Eventlist[Idx_Clu, 1],
                            s=1,
                            c=Cropped_Ch1_TightCluster_Eventlist[
                                Idx_Clu, ClusterUID_col
                            ],
                        )
                        ax.set_xlim(min_x, max_x)
                        ax.set_ylim(min_y, max_y)
                        ax.set_aspect("equal")
                        plt.tight_layout()

                        for ExportImage in ExportImageFormat:
                            lucky_fig_fname = os.path.join(
                                cropped_dir,
                                save_file_name
                                + " - Centre-Cropped Highlight Preview."
                                + ExportImage,
                            )
                            plt.savefig(
                                lucky_fig_fname,
                                dpi=600,
                                bbox_inches=0,
                                facecolor="none",
                                edgecolor="none",
                            )
                        plt.close()

                    # all points without colouring
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(111, facecolor="none")
                    plt.scatter(
                        Cropped_Ch1_TightCluster_Eventlist[:, 0],
                        Cropped_Ch1_TightCluster_Eventlist[:, 1],
                        s=1,
                        color="black",
                    )
                    ax.set_xlim(min_x, max_x)
                    ax.set_ylim(min_y, max_y)
                    ax.set_aspect("equal")
                    plt.tight_layout()

                    for ExportImage in ExportImageFormat:
                        lucky_fig_fname = os.path.join(
                            cropped_dir,
                            save_file_name
                            + " - Centre-Cropped Flat Preview."
                            + ExportImage,
                        )
                        plt.savefig(
                            lucky_fig_fname,
                            dpi=600,
                            bbox_inches=0,
                            facecolor="none",
                            edgecolor="none",
                            transparent=True,
                        )
                    plt.close()

                    plt.ion()

                # save the cropped table
                # data_headers from earlier save can be reused here
                data_fname = os.path.join(
                    cropped_dir, save_file_name + " - Cropped" + OutFileExt
                )
                np.savetxt(
                    data_fname,
                    Cropped_Ch1_TightCluster_Eventlist,
                    delimiter=OutDelimiter,
                    fmt="%10.3f",
                    header=data_headers,
                    comments="",
                )
        #
        # End of per-scenario processing

    # Save the Ground Truth Data as CSV to avoid colliding with TSV generated data
    #
    gnd_truth_headers = (
        "FileUID,"
        + "Points per um2,"
        + "Points per cluster,"
        + "Percent points clustered (target),"
        + "radius(min),"
        + "radius(max),"
        + "Clusters per um2,"
        + "Cell Repeat,"
        + "Cluster count (actual),"
        + "Clusters per um2 (actual),"
        + "Percent points clustered (actual),"
        + "Total points,"
        + "Total clustered points,"
        + "SimuCell Area (um2),"
        + "Points per um2 in RoI,"
        + "Points per um2 in Image"
    )

    np.savetxt(
        os.path.join(outputpath, "AAA Ground Truth.csv"),
        gnd_truth,
        delimiter=",",
        fmt="%10.5f",
        header=gnd_truth_headers,
        comments="",
    )

    #
    # Save a JSON file
    #
    if doGenerateJSON:
        # Create a JSON file to enable smooth procession to the next stage
        print("Saving JSON file...", end="", flush=True)
        if ImageSize[2] == 0:
            zCol = None
            zMinVal = None
            zMaxVal = None
        else:
            zCol = 3
            ClusterUID_col = 4
            zMinVal = 0
            zMaxVal = ImageSize[2].tolist()

        ps_json = {
            "xCol": 0,
            "yCol": 1,
            "zCol": zCol,
            "ClusMembershipIDCol": ClusterUID_col,
            "ChanIDCol": None,
            "UIDCol": None,
            "LabelIDCol": LabelID_col,
            "ImageSize": ImageSize.tolist(),
            "xMin": 0,
            "yMin": 0,
            "zMin": zMinVal,
            "xMax": ImageSize[0].tolist(),
            "yMax": ImageSize[1].tolist(),
            "zMax": zMaxVal,
            "InputFileDelimiter": "\t",
            "InputFileExt": ".tsv",
            "AutoAxes": False,
            "AutoAxesNearest": 1000,
            "ClosestFriend": int(NearestNeighbour),
            "FurthestFriend": int(FurthestNeighbour),
            "SaveImagesForRepeat": lucky_simRepeat,
        }

        json_fname = os.path.join(outputpath, "AAA Data Descriptions.json")
        with open(json_fname, "w") as file:
            file.write(json.dumps(ps_json, indent=4, sort_keys=True))
        print("done!")

    fn_etc.ok_msg("Finished generating data")
    print("The output folder was\t" + outputpath)
