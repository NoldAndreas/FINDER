import glob
import json
import os
import pickle
import sys
from datetime import datetime

# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pandas as pd
from Clustering import Clustering
from Definitions import data_folder
from Geometry import Geometry_Free, Geometry_Grid, Geometry_Path
from PlotScatter import plotScatter


def ComputeSeries(basefolder, input_filename, name_):
    """

    Parameters
    ----------
    basefolder:
        Folder in which the data are stored
    input_filename:
        Path to the input file
    name_: str
        name of the experiment
    """

    # load the parameters
    print("Loading parameters from " + input_filename + " ...")
    with open(input_filename, "r") as fp:
        params = json.load(fp)

    plot_option = True

    # ****************************
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")

    filename_base = "Results_" + name_
    filename_pickle = filename_base + ".pickle"
    filename_dataframe = filename_base + ".txt"
    filename_json = filename_base + "_Parameters.json"
    resultfolder = os.path.join(basefolder, "Results")
    basefolder_results = os.path.join(
        resultfolder, "Results_" + date_time + "_" + name_
    )

    # create `basefolder` if it does not exist
    if os.path.exists(basefolder):
        pass
    else:
        os.makedirs(basefolder)

    if os.path.exists(resultfolder):
        pass
    else:
        os.makedirs(resultfolder)

    # create the `results` folder if it does not exist
    if os.path.exists(basefolder_results):
        pass
    else:
        os.makedirs(basefolder_results)

    # store the parameter
    with open(os.path.join(basefolder_results, filename_json), "w") as fp:
        json.dump(params, fp, indent=5)

    data_eval = {
        params["var_1_name"]: [],
        params["var_2_name"]: [],
        "algo": [],
        "true_positives_ratio": [],
        "false_positives_ratio": [],
        "compute_time": [],
        "results": [],
    }

    var_1_name = params["var_1_name"]
    var_2_name = params["var_2_name"]

    def GetBaseName():
        return (
            basefolder_results
            + "/"
            + filename_base
            + "_"
            + var_1_name
            + "_"
            + str(params[var_1_name])
            + "_"
            + var_2_name
            + "_"
            + str(params[var_2_name])
        )

    for var_1 in params["var_1_values"]:
        params[var_1_name] = var_1

        for var_2 in params["var_2_values"]:
            params[var_2_name] = var_2
            print("Geometry is ", params["geometry"])
            # GEOMETRY == GRID
            if params["geometry"] == "grid":
                G = Geometry_Grid(
                    basefolder,
                    params["unit_type"],
                    n_side=params["n_side"],
                    Delta_ratio=params["Delta_ratio"],
                    noise_ratio=params["noise_ratio"],
                )
                G.GeneratePoints(params["seed"])

            # GEOMETRY == PATH

            elif params["geometry"] == "path":
                G = Geometry_Path(basefolder, params["unit_type"])
                if "noise_ratio" in params:
                    n_noise = (
                        params["noise_ratio"]
                        * G.GetTypical_Number_of_points_templateClusters()
                        * params["N_clusters"]
                    )
                    n_noise = np.int(n_noise)
                else:
                    n_noise = (
                        params["noise_ratio_per_cluster"]
                        * params["N_clusters"]
                    )
                G.GeneratePoints(params["N_clusters"], n_noise, params["seed"])

            # GEOMETRY == FREE
            elif params["geometry"] == "free":
                G = Geometry_Free(
                    basefolder,
                    params["unit_type"],
                    noise_ratio=params["noise_ratio"],
                )
                G.GeneratePoints(params["seed"])

                # Test: What does testset look like?
            if plot_option:
                plotScatter(
                    G.labels_groundtruth, G.XC, filename=GetBaseName() + ".pdf"
                )

            data_subcl = {"subcl": [], "algos": []}

            for algo in params["algos"]:
                # Step 2: Set up Object
                # print(basefolder)
                print()
                print("USING ", algo)
                print()
                CL = Clustering(G, basefolder)

                # Step 3: Compute clustering
                result_ = CL.fit(algo, params["params_algos"])
                CL.evaluate()

                np.savetxt(
                    basefolder_results
                    + "/"
                    + filename_base
                    + "labels_"
                    + algo
                    + ".txt",
                    CL.labels,
                    fmt="%.0f",
                )

                # Step 4a: Save figure
                if plot_option:
                    CL.plotScatter(GetBaseName() + "_algo_" + algo + ".pdf")

                # Step 4 b: Save result in pickle
                pickle_out = open(
                    basefolder_results + "/" + filename_pickle, "ab"
                )
                pickle.dump(CL, pickle_out)
                pickle_out.close()

                # Step 4 c: Save results in table
                data_eval[var_1_name].append(var_1)
                data_eval[var_2_name].append(var_2)
                data_eval["algo"].append(CL.algo)

                data_eval["compute_time"].append(CL.computationTime)
                data_eval["true_positives_ratio"].append(
                    CL.cluster_evaluation["true_positives"]
                    / CL.Geometry.N_clusters
                )
                data_eval["false_positives_ratio"].append(
                    CL.cluster_evaluation["false_positives"]
                    / CL.Geometry.N_clusters
                )
                data_eval["results"].append(result_)

                df = pd.DataFrame(data=data_eval)
                df.to_csv(basefolder_results + "/" + filename_dataframe)

                data_subcl["subcl"] += list(CL.number_of_subclusters)
                data_subcl["algos"] += [algo] * len(CL.number_of_subclusters)
            # Step 4 c: Save results in table

            df_subcl = pd.DataFrame(data=data_subcl)
            df_subcl.to_csv(
                basefolder_results
                + "/"
                + filename_base
                + "_subclusters"
                + str(var_1)
                + "_"
                + str(var_2)
                + ".txt"
            )


def ComputePhaseSpace(basefolder, input_filename, name_):
    # load the parameters
    print("Loading parameters from " + input_filename + " ...")
    with open(input_filename, "r") as fp:
        params = json.load(fp)

    plot_option = True

    # ****************************
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")

    filename_base = "\Results_" + name_
    filename_pickle = filename_base + ".pickle"
    filename_dataframe = filename_base + ".txt"
    filename_json = filename_base + "_Parameters.json"
    # basefolder_results = basefolder + "Results_" + date_time + "_" + name_ + "/"  #
    basefolder_results = os.path.join(
        basefolder, "Results_" + date_time + "_" + name_
    )

    # create `basefolder` if it does not exist
    if os.path.exists(basefolder):
        pass
    else:
        os.makedirs(basefolder)

    # create the `results` folder if it does not exist
    if os.path.exists(basefolder_results):
        pass
    else:
        os.makedirs(basefolder_results)

    # store the parameter
    with open(basefolder_results + filename_json, "w") as fp:
        json.dump(params, fp, indent=5)

    data_eval = {
        params["var_1_name"]: [],
        params["var_2_name"]: [],
        "algo": [],
        "true_positives_ratio": [],
        "false_positives_ratio": [],
        "compute_time": [],
        "results": [],
    }

    var_1_name = params["var_1_name"]
    var_2_name = params["var_2_name"]

    def GetBaseName():
        return (
            basefolder_results
            + filename_base
            + "_"
            + var_1_name
            + "_"
            + str(params[var_1_name])
            + "_"
            + var_2_name
            + "_"
            + str(params[var_2_name])
        )

    for var_1 in params["var_1_values"]:
        params[var_1_name] = var_1

        for var_2 in params["var_2_values"]:
            params[var_2_name] = var_2
            print("Geometry is ", params["geometry"])
            # GEOMETRY == GRID
            if params["geometry"] == "grid":
                G = Geometry_Grid(
                    basefolder,
                    params["unit_type"],
                    n_side=params["n_side"],
                    Delta_ratio=params["Delta_ratio"],
                    noise_ratio=params["noise_ratio"],
                )
                G.GeneratePoints(params["seed"])

            # GEOMETRY == PATH

            elif params["geometry"] == "path":
                G = Geometry_Path(basefolder, params["unit_type"])
                if "noise_ratio" in params:
                    n_noise = (
                        params["noise_ratio"]
                        * G.GetTypical_Number_of_points_templateClusters()
                        * params["N_clusters"]
                    )
                    n_noise = np.int(n_noise)
                else:
                    n_noise = (
                        params["noise_ratio_per_cluster"]
                        * params["N_clusters"]
                    )
                G.GeneratePoints(params["N_clusters"], n_noise, params["seed"])

            # GEOMETRY == FREE
            elif params["geometry"] == "free":
                G = Geometry_Free(
                    basefolder,
                    params["unit_type"],
                    noise_ratio=params["noise_ratio"],
                )
                G.GeneratePoints(params["seed"])

                # Test: What does testset look like?
            if plot_option:
                plotScatter(
                    G.labels_groundtruth, G.XC, filename=GetBaseName() + ".pdf"
                )

            data_subcl = {"subcl": [], "algos": []}

            for algo in params["algos"]:
                # Step 2: Set up Object
                # print(basefolder)
                print()
                print("USING ", algo)
                print()
                CL = Clustering(G, basefolder)

                # Step 3: Compute clustering
                result_ = CL.fit(algo, params["params_algos"])
                CL.evaluate()

                np.savetxt(
                    basefolder_results
                    + filename_base
                    + "labels_"
                    + algo
                    + ".txt",
                    CL.labels,
                    fmt="%.0f",
                )

                # Step 4a: Save figure
                if plot_option:
                    CL.plotScatter(GetBaseName() + "_algo_" + algo + ".pdf")

                # Step 4 b: Save result in pickle
                pickle_out = open(basefolder_results + filename_pickle, "ab")
                pickle.dump(CL, pickle_out)
                pickle_out.close()

                # Step 4 c: Save results in table
                data_eval[var_1_name].append(var_1)
                data_eval[var_2_name].append(var_2)
                data_eval["algo"].append(CL.algo)

                data_eval["compute_time"].append(CL.computationTime)
                data_eval["true_positives_ratio"].append(
                    CL.cluster_evaluation["true_positives"]
                    / CL.Geometry.N_clusters
                )
                data_eval["false_positives_ratio"].append(
                    CL.cluster_evaluation["false_positives"]
                    / CL.Geometry.N_clusters
                )
                data_eval["results"].append(result_)

                df = pd.DataFrame(data=data_eval)
                df.to_csv(basefolder_results + filename_dataframe)

                data_subcl["subcl"] += list(CL.number_of_subclusters)
                data_subcl["algos"] += [algo] * len(CL.number_of_subclusters)
            # Step 4 c: Save results in table

            df_subcl = pd.DataFrame(data=data_subcl)
            df_subcl.to_csv(
                basefolder_results
                + filename_base
                + "_subclusters"
                + str(var_1)
                + "_"
                + str(var_2)
                + ".txt"
            )


if __name__ == "__main__":

    basefolder = sys.argv[1]
    input_filename = sys.argv[2]
    name_ = input_filename[:-5] + "_" + sys.argv[3]

    # filenamesList  = glob.glob(basefolder+'Input/*.json')
    # print(filenamesList);

    #    input_filename = filenamesList[0];#"Fig3_b_4mers.json";
    # input_filename = os.path.basename(filenamesList[0]);
    # name_          = input_filename[:-5];

    if os.path.isfile(input_filename):
        pass
    else:
        input_filename = os.path.join(basefolder, "Input", input_filename)

    ComputeSeries(basefolder, input_filename, name_)
