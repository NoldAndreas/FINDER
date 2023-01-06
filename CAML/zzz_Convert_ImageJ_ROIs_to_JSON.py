#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:44:30 2019

@author: dave
"""

import json
import os

import pandas as pd
from natsort import natsorted

proc_wd = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print("Changed working directory to " + proc_wd)

import FuncEtc as fn_etc

inputpath_IJROIs = fn_etc.askforinput(
    message="Folder with ROI files saved from ImageJ (or FIJI)",
    errormessage="The folder you provided does not exist or you have supplied a file instead of a folder.",
    defaultval="",
    isvalid=lambda v: os.path.isdir(v),
)

outputpath_IJROIs = fn_etc.askforinput(
    message="Folder to save output JSON files",
    errormessage="An output folder name must be supplied!",
    defaultval=inputpath_IJROIs,
    isvalid=lambda v: len(v) > 0,
)

# check the starting_index value in case we are restarting a run
AskDoInvertY = fn_etc.askforinput(
    message="Invert the y-axis? (Y or N) ROIs from ImageJ selections will require y-axis inversion.",
    errormessage="Type Y or N",
    defaultval="N",
    isvalid=lambda v: v.lower() in ["y", "n", "yes", "no"],
)

if AskDoInvertY.lower() in ["y", "yes"]:

    doInvertYAxis = True

    AskMaxY = fn_etc.askforinput(
        message="Y-axis inversion: What is the extent of the Y-axis?",
        errormessage="Please supply a positive integer value.",
        defaultval="40960",
        isvalid=lambda v: v.isdigit() and int(v) >= 0,
    )
    MaxY = int(AskMaxY)

# make the folder for the output data
if not os.path.exists(outputpath_IJROIs):
    os.makedirs(outputpath_IJROIs)

# get a list of the input files
ij_roi_files = natsorted(
    [i for i in os.listdir(inputpath_IJROIs) if ".txt" in i]
)
total_ij_rois = len(ij_roi_files)

for fileIdx in range(0, total_ij_rois):

    current_file = os.path.join(inputpath_IJROIs, ij_roi_files[fileIdx])

    # load the data as an pandas dataframe
    roi_data = pd.read_csv(
        current_file,
        sep="\t",
        lineterminator="\n",
        usecols=[0, 1],
        header=None,
    )
    roi_data = roi_data.rename(
        columns={0: "ROIx", 1: "ROIy"}
    )  # not really necessary here, but just so we know what we have!

    if doInvertYAxis:
        roi_data["ROIy"] = MaxY - roi_data["ROIy"]

    roi_data_json = roi_data.to_dict(orient="split")["data"]

    outputfile = os.path.join(
        inputpath_IJROIs, current_file.split(".txt")[0] + ".json"
    )

    # save the data as a barebones JSON file
    with open(outputfile, "w") as outfile:
        json.dump(roi_data_json, outfile)

print(
    "-------------------------------------------------\n"
    + "\t\t\tCompleted!"
    + "\n-------------------------------------------------"
)
print("The input ImageJ ROIs folder was\t" + inputpath_IJROIs)
print("The output JSON ROIs folder was\t" + outputpath_IJROIs)
