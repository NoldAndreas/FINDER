#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a JSON file based on your answers to questions.

The JSON file is used to inform subsequent processing stages of the settings you want to use.

@author: dave
"""

import json
import os
import string

proc_wd = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print("Changed working directory to " + proc_wd)

import FuncEtc as fn_etc

if __name__ == "__main__":

    # Initial processing settings (ProcSettings) are loaded from a JSON file
    good_json = False

    while not good_json:

        # We will recycle any existing values, such as from previous attempts
        # to set things up, to save typing things out again

        TakenColumns = (
            []
        )  # a list to hold columns which have already been given by the user

        ##### file extension for all input data files #####

        if "InputFileExt" in locals():
            default_InputFileExt = InputFileExt
        else:
            default_InputFileExt = ".tsv"

        InputFileExt = fn_etc.askforinput(
            message="Input file extension",
            errormessage="Please enter a valid file extension.\nValid extensions have alphanumeric characters (a-z A-Z 0-9) and are preceeded by a period (.) character",
            defaultval=default_InputFileExt,
            isvalid=lambda v: (
                set(v) <= set(string.ascii_letters + string.digits + ".")
            )
            and v[0] == ".",
        )

        ##### File delimiter #####

        if "InputFileDelimiter" in locals():
            default_InputFileDelimiter = InputFileDelimiter
        else:
            default_InputFileDelimiter = "\\t"

        InputFileDelimiter = fn_etc.askforinput(
            message="Data delimiter - character or code which separates columns of data (for tab enter \\t)",
            errormessage="Please enter a valid delimiter.\nExample delimiters: \\t (tab) , (comma) or space characters\nPress Ctrl C to stop here.",
            defaultval=default_InputFileDelimiter,
            isvalid=lambda v: set(v)
            <= set(
                string.ascii_letters + " !\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
            ),
        )

        if InputFileDelimiter == "\\t":
            InputFileDelimiter = "\t"  # fix problem with escaped slashes on the tab delimiter case

        ##### Column for x-Coords #####

        if "xCol" in locals():
            default_xCol = str(xCol)
        else:
            default_xCol = "0"

        xCol = fn_etc.askforinput(
            message="x-coords column - Enter the column holding x-coordinates (remember: ZERO-based numbering!)",
            errormessage="An integer greater than zero is required.",
            defaultval=default_xCol,
            isvalid=lambda v: v.isdigit() and int(v) >= 0,
        )

        xCol = int(xCol)
        TakenColumns.append(xCol)

        ##### Column for y-Coords #####

        if "yCol" in locals():
            default_yCol = str(yCol)
        else:
            default_yCol = "1"

        yCol = fn_etc.askforinput(
            message="y-coords column - Enter the column holding y-coordinates (remember: ZERO-based numbering!)",
            errormessage="An integer greater than zero is required. You cannot specify the same column for multiple purposes (columns used: "
            + str(TakenColumns)
            + ")",
            defaultval=default_yCol,
            isvalid=lambda v: v.isdigit()
            and int(v) >= 0
            and int(v) not in TakenColumns,
        )

        yCol = int(yCol)

        TakenColumns.append(yCol)

        # z axis information

        ThreeDee = fn_etc.askforinput(
            message="Do you have a z-axis (3D data)?",
            errormessage="Enter yes or no",
            defaultval="no",
            isvalid=lambda v: v.lower() in ["y", "n", "yes", "no"],
        )

        if ThreeDee.lower() in ["y", "yes"]:

            ThreeDeeData = True

            ##### Column for z-Coords #####

            if "zCol" in locals():
                default_zCol = str(yCol)
            else:
                default_zCol = "2"

            zCol = fn_etc.askforinput(
                message="z-coords column - Enter the column holding z-coordinates (remember: ZERO-based numbering!)",
                errormessage="An integer greater than zero is required. You cannot specify the same column for multiple purposes (columns used: "
                + str(TakenColumns)
                + ")",
                defaultval=default_zCol,
                isvalid=lambda v: v.isdigit()
                and int(v) >= 0
                and int(v) not in TakenColumns,
            )

            zCol = int(zCol)

            TakenColumns.append(zCol)

        else:
            ThreeDeeData = False
            zCol = None

        ##### Column for Cluster Membership values #####

        if "ClusMembershipIDCol" in locals():
            default_ClusMembershipIDCol = str(ClusMembershipIDCol)
        else:
            default_ClusMembershipIDCol = "None"

        ClusMembershipIDCol = fn_etc.askforinput(
            message="Cluster Membership column - Enter the column holding Cluster Membership values (or None if there isn't one)",
            errormessage="An integer greater than zero is required or the word None. You cannot specify the same column for multiple purposes (columns used: "
            + str(TakenColumns)
            + ")",
            defaultval=default_ClusMembershipIDCol,
            isvalid=lambda v: (
                v.isdigit() and int(v) > 0 and int(v) not in TakenColumns
            )
            or v.lower() == "none",
        )

        if ClusMembershipIDCol.lower() == "none":
            ClusMembershipIDCol = None
        else:
            ClusMembershipIDCol = int(ClusMembershipIDCol)

        ##### Column for Imaging-Channel values #####

        if "ChanIDCol" in locals():
            default_ChanIDCol = str(ChanIDCol)
        else:
            default_ChanIDCol = "None"

        ChanIDCol = fn_etc.askforinput(
            message="Channel ID column - Enter the column holding Channel ID (or None if there isn't one)",
            errormessage="An integer greater than zero is required or the word None. You cannot specify the same column for multiple purposes (columns used: "
            + str(TakenColumns)
            + ")",
            defaultval=default_ChanIDCol,
            isvalid=lambda v: (
                v.isdigit() and int(v) > 0 and int(v) not in TakenColumns
            )
            or v.lower() == "none",
        )

        if ChanIDCol.lower() == "none":
            ChanIDCol = None
        else:
            ChanIDCol = int(ChanIDCol)

        ##### Column for Unique IDs #####

        if "UIDCol" in locals():
            default_UIDCol = str(UIDCol)
        else:
            default_UIDCol = "None"

        UIDCol = fn_etc.askforinput(
            message="UID column - Enter the column holding unique ID for each point (or None if there isn't one)",
            errormessage="An integer greater than zero is required or the word None. You cannot specify the same column for multiple purposes (columns used: "
            + str(TakenColumns)
            + ")",
            defaultval=default_UIDCol,
            isvalid=lambda v: (
                v.isdigit() and int(v) > 0 and int(v) not in TakenColumns
            )
            or v.lower() == "none",
        )

        if UIDCol.lower() == "none":
            UIDCol = None
        else:
            UIDCol = int(UIDCol)

        ##### Data Scake #####

        if "DataScale" in locals():
            default_DataScale = str(DataScale)
        else:
            default_DataScale = "1"

        DataScale = fn_etc.askforinput(
            message="How are your xy data scaled? Enter a value as 'nanometers per data-unit'.",
            errormessage="An integer greater than zero is required.",
            defaultval=default_DataScale,
            isvalid=lambda v: v.isdigit() and int(v) > 0,
        )

        DataScale = int(DataScale)

        ##### Automatically determine the axis limits #####

        if "AutoAxes" in locals():
            if AutoAxes:
                default_AutoAxes = "Yes"
            else:
                default_AutoAxes = "No"
        else:
            default_AutoAxes = "No"

        AutoAxes = fn_etc.askforinput(
            message="Do you want to estimate the x- and y-axis ranges for each image?",
            errormessage="Enter yes or no",
            defaultval=default_AutoAxes,
            isvalid=lambda v: v.lower() in ["y", "n", "yes", "no"],
        )

        if AutoAxes.lower() in ["y", "yes"]:
            AutoAxes = True
        else:
            AutoAxes = False

        ##### If using AutoAxes we can skip some questions (otherwise we need more information)) #####

        if AutoAxes:

            # fixed values for these entries (they will be ignored with AutoAxes on)
            xMin = 0
            xMax = 40000
            yMin = 0
            yMax = 40000
            ImageSize = [40000, 40000]

            ##### Round Axis limits to the nearest value #####

            if "AutoAxesNearest" in locals():
                default_AutoAxesNearest = str(AutoAxesNearest)
            else:
                default_AutoAxesNearest = "1000"

            AutoAxesNearest = fn_etc.askforinput(
                message="Round x- and y-axis range estimates to the nearest [what] nm?",
                errormessage="An integer greater than zero is required.",
                defaultval=default_AutoAxesNearest,
                isvalid=lambda v: v.isdigit() and int(v) > 0,
            )

        else:

            AutoAxesNearest = 1000

            ##### x Minimum #####

            if "xMin" in locals():
                default_xMin = str(xMin)
            else:
                default_xMin = "0"

            xMin = fn_etc.askforinput(
                message="Minimum x-coordinate value",
                errormessage="An integer is required.",
                defaultval=default_xMin,
                isvalid=lambda v: v.isdigit(),
            )

            xMin = int(xMin)

            ##### x Maximum #####

            if "xMax" in locals():
                default_xMax = str(xMax)
            else:
                default_xMax = "40960"

            xMax = fn_etc.askforinput(
                message="Maximum x-coordinate value",
                errormessage="An integer is required and it must be greater than the x-minimum of "
                + str(xMin),
                defaultval=default_xMax,
                isvalid=lambda v: v.isdigit() and int(v) > xMin,
            )

            xMax = int(xMax)

            ##### y Minimum #####

            if "yMin" in locals():
                default_yMin = str(yMin)
            else:
                default_yMin = "0"

            yMin = fn_etc.askforinput(
                message="Minimum y-coordinate value",
                errormessage="An integer is required.",
                defaultval=default_yMin,
                isvalid=lambda v: v.isdigit(),
            )

            yMin = int(yMin)

            ##### y Maximum #####

            if "yMax" in locals():
                default_yMax = str(yMax)
            else:
                default_yMax = "40960"

            yMax = fn_etc.askforinput(
                message="Maximum y-coordinate value",
                errormessage="An integer is required and it must be greater than the y-minimum of "
                + str(yMin),
                defaultval=default_yMax,
                isvalid=lambda v: v.isdigit() and int(v) > yMin,
            )

            yMax = int(yMax)

            if ThreeDeeData:

                ##### z Minimum #####

                if "zMin" in locals():
                    default_zMin = str(zMin)
                else:
                    default_zMin = "0"

                zMin = fn_etc.askforinput(
                    message="Minimum z-coordinate value",
                    errormessage="An integer is required.",
                    defaultval=default_zMin,
                    isvalid=lambda v: v.isdigit(),
                )

                zMin = int(zMin)

                ##### z Maximum #####

                if "zMax" in locals():
                    default_zMax = str(zMax)
                else:
                    default_zMax = "1000"

                zMax = fn_etc.askforinput(
                    message="Maximum z-coordinate value",
                    errormessage="An integer is required and it must be greater than the z-minimum of "
                    + str(yMin),
                    defaultval=default_yMax,
                    isvalid=lambda v: v.isdigit() and int(v) > zMin,
                )

                zMax = int(zMax)

                ImageSize = [xMax - xMin, yMax - yMin, zMax - zMin]

            else:

                zMin = None
                zMax = None

                ImageSize = [xMax - xMin, yMax - yMin, 0]

        #            ##### Image size #####
        #
        #            if 'ImageSize' in locals():
        #                default_ImageSize = str(ImageSize)
        #            else:
        #                default_ImageSize = '40960'
        #
        #            ImageSize = fn_etc.askforinput(
        #                message = 'Image size in nanometers for each dimension, as [x, y] or [x, y, z]',
        #                errormessage= 'An integer greater than zero is required.',
        #                defaultval= default_ImageSize,
        #                isvalid = lambda v: v.isdigit() and int(v) > 0)
        #
        #            ImageSize = int(ImageSize)

        ##### ClosestFriend #####

        if "ClosestFriend" in locals():
            default_ClosestFriend = str(ClosestFriend)
        else:
            default_ClosestFriend = "1"

        ClosestFriend = fn_etc.askforinput(
            message="Begin measuring distances from this nth nearest-neighbour (Closest Near-Neighbour, does not include self)",
            errormessage="An integer greater than zero is required",
            defaultval=str(default_ClosestFriend),
            isvalid=lambda v: v.isdigit() and int(v) > 0,
        )

        ClosestFriend = int(ClosestFriend)

        ##### FurthestFriend #####

        if "FurthestFriend" in locals():
            default_FurthestFriend = str(FurthestFriend)
        else:
            default_FurthestFriend = "1000"

        FurthestFriend = fn_etc.askforinput(
            message="Finish measuring distances with this nth nearest-neighbour (Furthest Near-Neighbour)",
            errormessage="An integer is required and must be greater than "
            + str(ClosestFriend)
            + " (Closest Near-Neighbour from which distances are measured)",
            defaultval=default_FurthestFriend,
            isvalid=lambda v: v.isdigit() and int(v) > ClosestFriend,
        )

        FurthestFriend = int(FurthestFriend)

        #### Confirm Settings ####

        fn_etc.info_msg("Imported JSON variables:")
        print(" │")
        print(" ├─InputFileExt:\t" + InputFileExt)
        print(" ├─InputFileDelimiter:\t" + InputFileDelimiter)
        print(" │")
        print(" ├─xCol:\t\t" + str(xCol))
        print(" ├─yCol:\t\t" + str(yCol))
        if ThreeDeeData:
            print(" ├─zCol:\t\t" + str(zCol))
        print(" ├─ClusMembershipIDCol:\t" + str(ClusMembershipIDCol))
        print(" ├─ChanIDCol:\t\t" + str(ChanIDCol))
        print(" ├─UIDCol:\t\t" + str(UIDCol))
        print(" │")
        print(" ├─DataScale:\t\t" + str(DataScale))
        print(" ├─AutoAxes:\t\t" + str(AutoAxes))
        if AutoAxes:
            print(" ├─AutoAxesNearest:\t" + str(AutoAxesNearest))
            print(" ├─ImageSize:\t\tWill be determined by AutoAxes")
            print(" ├─xMin:\t\tWill be determined by AutoAxes")
            print(" ├─xMax:\t\tWill be determined by AutoAxes")
            print(" ├─yMin:\t\tWill be determined by AutoAxes")
            print(" ├─yMax:\t\tWill be determined by AutoAxes")
            if ThreeDeeData:
                print(" ├─zMin:\t\tWill be determined by AutoAxes")
                print(" ├─zMax:\t\tWill be determined by AutoAxes")
        else:
            print(" ├─AutoAxesNearest:\tNo (not used)")
            print(" ├─ImageSize:\t\t" + str(ImageSize))
            print(" ├─xMin:\t\t" + str(xMin))
            print(" ├─xMax:\t\t" + str(xMax))
            print(" ├─yMin:\t\t" + str(yMin))
            print(" ├─yMax:\t\t" + str(yMax))
            if ThreeDeeData:
                print(" ├─zMin:\t\t" + str(zMin))
                print(" ├─zMax:\t\t" + str(zMax))
        print(" │")
        print(" ├─ClosestFriend:\t" + str(ClosestFriend))
        print(" └─FurthestFriend:\t" + str(FurthestFriend))

        verify_good_json = fn_etc.askforinput(
            message="Are these settings correct? Enter 'Y' to proceed or enter 'N' to respecify your JSON file's settings",
            errormessage="Type Y or N",
            defaultval="y",
            isvalid=lambda v: v.lower() in ["y", "n", "yes", "no"],
        )

        if verify_good_json.lower() in ["y", "yes"]:
            print("JSON file accepted.")
            good_json = True
        else:
            print(
                "Settings rejected. Please give new settings or Ctrl X to cancel now."
            )

    #
    ## End of specifying variables for JSON file
    #

    ##### JSON save folder #####

    if "JSON_out_folder" in locals():
        default_JSON_out_folder = JSON_out_folder
    else:
        default_JSON_out_folder = ""

    JSON_out_folder = fn_etc.askforinput(
        message="FOLDER to save the JSON file (full path)",
        errormessage="The folder you provided does not exist or you supplied the path to a file. Check the path and try again.",
        defaultval=default_JSON_out_folder,
        isvalid=lambda v: os.path.isdir(v),
    )

    ##### JSON file name #####

    if "JSON_out_file" in locals():
        default_JSON_out_file = JSON_out_file
    else:
        default_JSON_out_file = "AAA - MyData.json"

    JSON_out_file = fn_etc.askforinput(
        message="NAME to give JSON file",
        errormessage="Please enter a valid file name.\nValid names may include alphanumeric characters (a-z A-Z 0-9)and special characters space ( ), period (.), underscore (_), and minus-hyphen (-).",
        defaultval=default_JSON_out_file,
        isvalid=lambda v: set(v)
        <= set(string.ascii_letters + string.digits + " ._-"),
    )

    if not JSON_out_file.endswith(".json"):
        JSON_out_file += ".json"

    # Create a JSON file to enable smooth procession to the next stage
    JSONsettings = {
        "InputFileExt": InputFileExt,
        "InputFileDelimiter": InputFileDelimiter,
        "xCol": xCol,
        "yCol": yCol,
        "ClusMembershipIDCol": ClusMembershipIDCol,
        "ChanIDCol": ChanIDCol,
        "UIDCol": UIDCol,
        "DataScale": DataScale,
        "AutoAxes": AutoAxes,
        "AutoAxesNearest": AutoAxesNearest,
        "ImageSize": ImageSize,
        "xMin": xMin,
        "xMax": xMax,
        "yMin": yMin,
        "yMax": yMax,
        "ClosestFriend": ClosestFriend,
        "FurthestFriend": FurthestFriend,
    }

    zzz_json_fname = os.path.join(JSON_out_folder, JSON_out_file)
    with open(zzz_json_fname, "w") as jsonfile:
        jsonfile.write(json.dumps(JSONsettings, indent=4, sort_keys=True))

    fn_etc.ok_msg("JSON file has been saved to:  " + zzz_json_fname)
    print(
        "-------------------------------------------------\n\tCompleted!\n-------------------------------------------------"
    )
