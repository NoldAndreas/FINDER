#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accept files created by 1.0_Data_Preparation and pools points into a set
for training and validating new models.

@author: dave

"""

import gc
import json
import os
import pickle
import time
from random import shuffle

import numpy as np
import pandas as pd
from natsort import natsorted

proc_wd = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != proc_wd:
    os.chdir(proc_wd)
    print("Changed working directory to " + proc_wd)

import FuncEtc as fn_etc

# ========================================================================
#  About the training data
# ========================================================================

total_features = 1  # how many input features are we tracking,
# e.g. 1=distances, 2 = x,y coords, 3 = x,y,z coords,

total_labels = 2  # how many classification labels can we apply?
# e.g. 2=clustered/notclustered

# a default list for two labels:
default_label_names = ["Non-clustered", "Clustered"]

## a default list for three labels:
# default_label_names = ['Non-clustered', 'Clustered (round)', 'Clustered (fibre)']

feature_type = (
    "dists"  # Either 'dists' or 'coords' to train on the distance values
)
# or the xy(z) coordinates.
# if 'coords' then you want to check that total_features is
# 2 or 3 for 2D or 3D images.


# ========================================================================
#  Harvesting points from images
# ========================================================================
# from each input image, how many points of each class will be added to the pool?
points_per_label = (
    2000  # This many points (of each label) will be taken per image.
)
# If this many points (of a label) are not available then
# as many as are available will be taken.
# If no points (of a label) are available then no points
# will be taken to the pool, for that image.
#
# ... OR ...
#
use_all_points = True  # if this is True then all points (of all labels) will
# be taken from each image. Useful if you have limited
# amount of training data or your labels are not evenly
# found within your source images.


# ========================================================================
#  Total size of points-pools
# ========================================================================
# Overall number of points to use for training, validation, and testing datasets
total_training_count = 500000  # default 500000
total_validation_count = 100000  # default 100000
total_testing_count = 100000  # default 100000

# ========================================================================
#  Composition of each pool
# ========================================================================
# Each pool of points will have a mixture of the available labels.
even_label_ratios = True  # If True each label will be equally represented.
# If False then you must specify your own
# custom mixture of labels, below.

# Customised mixture of the labels within the training datasets.
# These three lines will be *ignored* if you are using even_label_ratios=True
# - Ensure there is an entry for each label!
# - Ensure the sum of all entries = 1.0!

# Two labels - custom mix of labels (if not using even_label_ratios, above)
training_label_fraction = [0.65, 0.35]
validation_label_fraction = [0.45, 0.55]
testing_label_fraction = [0.1, 0.9]

## Three labels - custom mix of labels (if not using even_label_ratios, above)
# training_label_fraction = [0.5, 0.35, 0.15]
# validation_label_fraction = [0.25, 0.5, 0.25]
# testing_label_fraction = [0.1, 0.1, 0.8]

# ========================================================================
#  End of user-editable variables
# ========================================================================

if __name__ == "__main__":

    # assuming you just ran the first stage, we can recycle the output folder from
    # that stage as the input for this stage.
    default_preplog = ""
    default_input_path = os.getcwd()

    if "s1_prep_outputpath" in locals():
        # output exists from previous 1_DataPreparation script
        default_input_path = s1_prep_outputpath

    inputpath_train = fn_etc.askforinput(
        message="Folder with distance measurements from "
        + "Stage 1 (.MemMap and .json etc)",
        errormessage="The folder you provided does not exist",
        defaultval=default_input_path,
        isvalid=lambda v: os.path.isdir(v),
    )

    if os.path.exists(os.path.join(inputpath_train, "PrepLog.txt")):
        default_preplog = os.path.join(inputpath_train, "PrepLog.txt")
        print("")  # blank line
        fn_etc.info_msg("Found PrepLog.txt in the input folder")
        PrepLogMessage = "Confirm the path to PrepLog.txt file"
    else:
        PrepLogMessage = "Enter the path to PrepLog.txt file"

    input_PrepLog = fn_etc.askforinput(
        message=PrepLogMessage,
        errormessage="The file you provided does not exist",
        defaultval=default_preplog,
        isvalid=lambda v: os.path.isfile(v),
    )

    del PrepLogMessage

    if os.path.isfile(input_PrepLog):
        PrepLog_imported = pd.read_csv(input_PrepLog, delimiter="\t")
        fn_etc.info_msg("Loaded PrepLog file from:")
        fn_etc.info_msg(input_PrepLog)

        # Clean any stray header lines that might have come in after restarting some processing
        if PrepLog_imported.FileID.dtype != "int64":
            PrepLog_imported = PrepLog_imported[
                ~PrepLog_imported.FileID.str.contains("FileID")
            ]
            PrepLog_imported = PrepLog_imported.apply(
                pd.to_numeric
            )  # convert to numeric (rather than object)
            PrepLog_imported = PrepLog_imported.reset_index(
                drop=True
            )  # re-index to ignore the missing row(s)
    else:
        fn_etc.err_msg("Cannot find PrepLog.txt in the specified location")
        raise ValueError(
            "Cannot find PrepLog.txt! "
            + "has it moved since you gave its location?"
        )

    s2_trnvaltest_outputpath = fn_etc.askforinput(
        message="Location of the output folder (will be created if it doesn't exist)",
        errormessage="The dataset must be named",
        defaultval=os.path.abspath(
            os.path.join(inputpath_train, "..", "2_training_pool")
        ),
        isvalid=lambda v: len(v) > 0 and not v.isspace(),
    )

    # Confirm the human-readable labels to use in plots and messages.
    label_names = []  # will hold the human-readable names for all our labels
    label_counters = []  # will hold the counts for each label

    for label in range(total_labels):

        try:
            suggested_label_name = default_label_names[label]
        except:
            suggested_label_name = "Type " + str(label)

        label_name = fn_etc.askforinput(
            message="Human-readable name for points with Label "
            + str(label)
            + "",
            errormessage="The label must have a name! The name must also have at least one readable character in it.",
            defaultval=suggested_label_name,
            isvalid=lambda v: len(v) > 0 and not v.isspace(),
        )

        label_names.append(str(label_name))
        label_counters.append(0)

    ## Either calculate the label ratios or check that the custom ratios are good.
    if even_label_ratios:

        label_split = 1 / total_labels

        training_label_fraction = []
        validation_label_fraction = []
        testing_label_fraction = []

        for label in range(total_labels):
            training_label_fraction.append(label_split)
            validation_label_fraction.append(label_split)
            testing_label_fraction.append(label_split)
    else:
        # customized split of the labels detected!
        # Check that enough custom-splits have been supplied
        if len(training_label_fraction) != total_labels:
            raise ValueError(
                "You have given "
                + str(total_labels)
                + " labels but only supplied "
                + str(len(training_label_fraction))
                + " entries for training_label_fraction!"
            )
        if len(validation_label_fraction) != total_labels:
            raise ValueError(
                "You have given "
                + str(total_labels)
                + " labels but only supplied "
                + str(len(validation_label_fraction))
                + " entries for training_label_fraction!"
            )
        if len(testing_label_fraction) != total_labels:
            raise ValueError(
                "You have given "
                + str(total_labels)
                + " labels but only supplied "
                + str(len(testing_label_fraction))
                + " entries for training_label_fraction!"
            )
        # Check that each arrangment adds up to 1
        if sum(training_label_fraction) != 1:
            raise ValueError(
                "The values for 'training_label_fraction' must sum up to 1.0"
            )
        if sum(validation_label_fraction) != 1:
            raise ValueError(
                "The values for 'validation_label_fraction' must sum up to 1.0"
            )
        if sum(testing_label_fraction) != 1:
            raise ValueError(
                "The values for 'testing_label_fraction' must sum up to 1.0"
            )

    ## get a list of the files to process from the given folder

    # mmap array for the distances to the Nth NNs.
    dist_files_train = natsorted(
        [
            i
            for i in os.listdir(inputpath_train)
            if os.path.isfile(os.path.join(inputpath_train, i))
            and "Dists.MemMap" in i
        ]
    )

    # mmap array for the original xy input data
    data_files_train = natsorted(
        [
            i
            for i in os.listdir(inputpath_train)
            if os.path.isfile(os.path.join(inputpath_train, i))
            and "Data.MemMap" in i
        ]
    )

    # mmap array for the 'target_vectors' classifier for training data
    target_vectors_files_train = natsorted(
        [
            i
            for i in os.listdir(inputpath_train)
            if os.path.isfile(os.path.join(inputpath_train, i))
            and "TargetVectors.MemMap" in i
        ]
    )

    # mmap array for the 'target_binary' classifier for training data
    target_binary_files_train = natsorted(
        [
            i
            for i in os.listdir(inputpath_train)
            if os.path.isfile(os.path.join(inputpath_train, i))
            and "TargetBinary.MemMap" in i
        ]
    )

    # mmap array for the 'target_labels' classifier for training data
    target_labels_files_train = natsorted(
        [
            i
            for i in os.listdir(inputpath_train)
            if os.path.isfile(os.path.join(inputpath_train, i))
            and "TargetLabels.MemMap" in i
        ]
    )

    # json file describing the metadata for each input file (image size, NN etc)
    json_files_train = natsorted(
        [
            i
            for i in os.listdir(inputpath_train)
            if os.path.isfile(os.path.join(inputpath_train, i))
            and ".json" in i
        ]
    )

    total_files_train = np.shape(dist_files_train)[0]

    if total_files_train > 0:

        if total_files_train != np.shape(PrepLog_imported)[0]:
            ErrorMessage = (
                "Input folder contains "
                + str(total_files_train)
                + " files but your PrepLog describes "
                + str(np.shape(PrepLog_imported)[0])
                + " files!"
            )
            fn_etc.err_msg(
                "Files in the input folder don't align with those in the PrepLog.txt file."
            )
            raise ValueError(ErrorMessage)
        else:
            if (
                total_files_train
                == np.shape(data_files_train)[0]
                == np.shape(json_files_train)[0]
                == np.shape(target_vectors_files_train)[0]
                == np.shape(target_binary_files_train)[0]
            ):
                fn_etc.info_msg(
                    "Found "
                    + str(total_files_train)
                    + " datasets to work with..."
                )
            else:
                if (
                    np.shape(target_labels_files_train)[0]
                    == np.shape(target_vectors_files_train)[0]
                    == np.shape(target_binary_files_train)[0]
                    == 0
                ):
                    ErrorMessage = "There appears to be no useable files for model training. Cannot find TargetLabels.MemMap files!"
                    fn_etc.err_msg("Missing required files for training!")
                else:
                    ErrorMessage = "For training and validation, each dataset requires a corresponding set of four MemMap files (Dists, Data, TargetBinary, and TargetVectors) plus a .json file from the first step (DataPreparation) to proceed.\nYou will see this error if you supply input data which was prepared for evaluation rather than for training."
                    fn_etc.err_msg("Mismatched input files!")

                print(
                    " JSON describes "
                    + str(np.shape(json_files_train)[0])
                    + " files to process. All sets must match for training to proceed."
                )
                print(" ├─Dists files:\t\t" + str(total_files_train))
                print(
                    " ├─Data files:\t\t" + str(np.shape(data_files_train)[0])
                )
                print(
                    " ├─TargetBinary files:\t"
                    + str(np.shape(target_binary_files_train)[0])
                )
                print(
                    " └─TargetLabels files:\t"
                    + str(np.shape(target_labels_files_train)[0])
                )
                print(
                    " └─TargetVectors files:\t"
                    + str(np.shape(target_vectors_files_train)[0])
                )

                raise ValueError(ErrorMessage)

    else:
        fn_etc.err_msg("No files to process!")
        raise ValueError(
            "No files to process! Check you have given the correct folder..."
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

    # create the output folder
    if not os.path.exists(s2_trnvaltest_outputpath):
        os.makedirs(s2_trnvaltest_outputpath)

    # containers to hold the pooled binary and vector labels
    vectors_pooled = np.empty((0, 1))
    binaries_pooled = np.empty((0, 1))
    labels_pooled = np.empty((0, 1))

    shape_tracker = (
        0  # to enable growing of our mmap array as new data comes in.
    )
    # We can't specify the size of this in advance because we can't be certain how
    # many points we will be able to extract from the raw training files.

    for fileIdx in range(0, total_files_train):

        current_file = json_files_train[fileIdx]

        # load this file's ProcSettings from json
        with open(os.path.join(inputpath_train, current_file), "r") as file:
            ps_current = json.loads(file.read())

        if fileIdx == 0:
            # only do this the first time around
            #            guess_size = np.array((points_per_label * total_labels * total_files_train, ps_current['FurthestFriend'], total_features))
            #            print('Guessing mmapsize')
            points_pool_fname = os.path.join(
                s2_trnvaltest_outputpath, "train_pool.MemMap"
            )
            points_pool_mmap = np.memmap(
                points_pool_fname,
                dtype="float64",
                shape=((3, ps_current["FurthestFriend"])),
                mode="w+",
            )

        fn_etc.info_msg(
            str(fileIdx + 1)
            + "/"
            + str(total_files_train)
            + " - "
            + ps_current["FilePrefix"]
        )

        # load distances from memmap file
        import_dists_f = os.path.join(
            inputpath_train, ps_current["dists_mmap_fname"]
        )
        if os.path.isfile(import_dists_f):
            Dists_all_New = np.memmap(
                import_dists_f,
                dtype="float64",
                shape=tuple(ps_current["DistsDumpShape"]),
                mode="r",
            )

        # load the target_vectors mmap file.
        import_vectors_f = os.path.join(
            inputpath_train, ps_current["target_vectors_mmap_fname"]
        )
        if os.path.isfile(import_vectors_f):
            target_vectors = np.memmap(
                import_vectors_f,
                dtype="int",
                shape=tuple(ps_current["VectorsDumpShape"]),
                mode="r",
            )

        # load the target_binary mmap file.
        import_binary_f = os.path.join(
            inputpath_train, ps_current["target_binary_mmap_fname"]
        )
        if os.path.isfile(import_binary_f):
            target_binary = np.memmap(
                import_binary_f,
                dtype="int",
                shape=tuple(ps_current["BinaryDumpShape"]),
                mode="r",
            )

        # load the target_labels mmap file.
        import_labels_f = os.path.join(
            inputpath_train, ps_current["target_labels_mmap_fname"]
        )
        if os.path.isfile(import_labels_f):
            target_labels = np.memmap(
                import_labels_f,
                dtype="int",
                shape=tuple(ps_current["LabelsDumpShape"]),
                mode="r",
            )

        # load the original coordinate data from mmap
        import_datatable_f = os.path.join(
            inputpath_train, ps_current["datatable_mmap_fname"]
        )
        if os.path.isfile(import_datatable_f):
            target_datatable = np.memmap(
                import_datatable_f,
                dtype="float64",
                shape=tuple(ps_current["DataDumpShape"]),
                mode="r",
            )

        TotalPointsThisImage = Dists_all_New.shape[0]
        # total_type0_points_this_file = np.sum(target_labels == 0)
        total_clustered_points_this_file = np.sum(target_binary)

        # sanity check it still matches with the equivalent line in PrepLog
        if (
            TotalPointsThisImage == PrepLog_imported["Total Points"][fileIdx]
            and total_clustered_points_this_file
            == PrepLog_imported["Clustered Points"][fileIdx]
        ):
            print("\t- " + str(TotalPointsThisImage) + " points:")
        else:
            ErrorMessage = "Files are not in order!"
            fn_etc.err_msg(ErrorMessage)
            raise ValueError(ErrorMessage)

        training_indices = np.empty(
            (0,), dtype=int
        )  # initialize the training indices with a dummy row
        for label in range(total_labels):

            # find the indices of points with our current target label
            label_indices = np.where(target_labels == label)[0]

            # choose random points with our target label (according to points_per_label)
            # if we opted to use all the points in the image then we don't bother
            # shuffling and choosing random ones
            if label_indices.shape[0] > 0:
                if not use_all_points:
                    shuffle(label_indices)
                    label_indices = label_indices[0:points_per_label]

                label_counters[label] = (
                    label_counters[label] + np.shape(label_indices)[0]
                )

            print(
                "\t- Harvested: \t"
                + str(np.shape(label_indices)[0])
                + "\t "
                + label_names[label]
                + " points."
            )

            training_indices = np.hstack((training_indices, label_indices))

        # shuffle the collection of indices again to mix up the occurance of each label
        shuffle(training_indices)

        total_retained_points = np.shape(training_indices)[0]

        # make a new memmap, copied from the existing pool, enlarged to hold the new points.
        new_mmap = np.memmap(
            points_pool_fname,
            mode="r+",
            dtype="float64",
            shape=(
                shape_tracker + total_retained_points,
                ps_current["FurthestFriend"],
                total_features,
            ),
        )

        # put the new points into the larger pool
        if feature_type == "dists":

            new_mmap[
                shape_tracker : shape_tracker + total_retained_points, :
            ] = Dists_all_New[training_indices, :, 0, None]

        elif feature_type == "coords":

            # get the 'origin' coordinates for the training points
            if ps_current["three_dee"]:
                training_origin_pts = target_datatable[training_indices, :3]
            else:
                training_origin_pts = target_datatable[training_indices, :2]

            # the Distances table includes all the data for each neighbour from the original data table.
            # As the first column of Dists_all_New is the distance values, we can add 1 to get the other types of data like x, y etc.
            training_NN_x = Dists_all_New[
                training_indices, :, ps_current["xCol"] + 1
            ]
            training_NN_y = Dists_all_New[
                training_indices, :, ps_current["yCol"] + 1
            ]

            # append the origin-point x and y coords at the end of the NN list (it will be removed later)
            training_NN_x = np.concatenate(
                (training_NN_x, training_origin_pts[:, 0, None]), axis=1
            )
            training_NN_y = np.concatenate(
                (training_NN_y, training_origin_pts[:, 1, None]), axis=1
            )

            # subtract the xy origin coords from each and normalize (check the appended origin should all be zero)
            training_NN_xcoords_norm = (
                training_NN_x - training_origin_pts[:, 0, None]
            )
            training_NN_ycoords_norm = (
                training_NN_y - training_origin_pts[:, 1, None]
            )

            # normalize each row internally to it's min/max, including the origin point coords
            row_x_mins = np.min(training_NN_xcoords_norm, axis=1)
            row_x_maxs = np.max(training_NN_xcoords_norm, axis=1)
            row_x_ranges = row_x_maxs - row_x_mins
            training_NN_xcoords_norm = (
                training_NN_xcoords_norm - row_x_mins[:, None]
            ) / row_x_ranges[:, None]
            training_NN_xcoords_norm = (
                training_NN_xcoords_norm
                - training_NN_xcoords_norm[:, -1, None]
            )  # recenter about the origin point x-coord

            row_y_mins = np.min(training_NN_ycoords_norm, axis=1)
            row_y_maxs = np.max(training_NN_ycoords_norm, axis=1)
            row_y_ranges = row_y_maxs - row_y_mins
            training_NN_ycoords_norm = (
                training_NN_ycoords_norm - row_y_mins[:, None]
            ) / row_y_ranges[:, None]
            training_NN_ycoords_norm = (
                training_NN_ycoords_norm
                - training_NN_ycoords_norm[:, -1, None]
            )  # recenter about the origin point y-coord

            #            # plot this point's neighbours
            #            plt.figure().add_subplot(111, aspect='equal').scatter(training_NN_xcoords_norm[1,:], training_NN_ycoords_norm[1,:])
            #            plt.show()

            # put the normalized coords into the training pool, except for the origin point (appended to the end, earlier, and should be all zeros)
            new_mmap[
                shape_tracker : shape_tracker + total_retained_points, :, 0
            ] = training_NN_xcoords_norm[:, :-1]
            new_mmap[
                shape_tracker : shape_tracker + total_retained_points, :, 1
            ] = training_NN_ycoords_norm[:, :-1]

            if ps_current["three_dee"]:
                # do the above for the z data
                training_NN_z = Dists_all_New[
                    training_indices, :, ps_current["zCol"] + 1
                ]
                training_NN_z = np.concatenate(
                    (training_origin_pts[:, 2, None], training_NN_z), axis=1
                )
                training_NN_zcoords_norm = (
                    training_NN_z - training_origin_pts[:, 2, None]
                )

                row_z_mins = np.min(training_NN_zcoords_norm, axis=1)
                row_z_maxs = np.max(training_NN_zcoords_norm, axis=1)
                row_z_ranges = row_z_maxs - row_z_mins
                training_NN_zcoords_norm = (
                    training_NN_zcoords_norm - row_z_mins[:, None]
                ) / row_z_ranges[:, None]
                training_NN_zcoords_norm = (
                    training_NN_zcoords_norm
                    - training_NN_zcoords_norm[:, -1, None]
                )  # recenter about the origin point z-coord

                new_mmap[
                    shape_tracker : shape_tracker + total_retained_points, :, 1
                ] = training_NN_zcoords_norm[:, :-1]
        else:
            raise ValueError(
                "Can only work with feature_type set to 'dists' or 'coords'"
            )

        # replace the older pool with the expanded pool
        points_pool_mmap = new_mmap

        # update the shape of the pool
        shape_tracker = shape_tracker + total_retained_points

        # pool the classifications too
        vectors_pooled = np.vstack(
            (vectors_pooled, target_vectors[training_indices])
        )
        binaries_pooled = np.vstack(
            (binaries_pooled, target_binary[training_indices])
        )
        labels_pooled = np.vstack(
            (labels_pooled, target_labels[training_indices])
        )

        # flush these before the next round
        training_indices = None
        label_indices = None

        _ = gc.collect()
        print(
            "\t- Added points to pool. "
            + str(shape_tracker)
            + " points pooled in total."
        )
        # fn_etc.complete_msg

    # end of for each fileIdx loop

    del new_mmap

    # Collect the indices of the different labels
    pool_all_labels_indices = (
        []
    )  # this list will hold a list of point-indices for each label
    for label in range(total_labels):
        print(
            "\tShuffling pooled points for " + label_names[label] + "... ",
            end="",
        )
        pool_label_indices = np.where(labels_pooled == label)[0]
        shuffle(pool_label_indices)
        pool_all_labels_indices.append(pool_label_indices.tolist())

        print("\x1b[36;1m\t" + "OK!" + "\x1b[0m", flush=True)
        # print(len(pool_all_labels_indices)) # length will be same as number of labels given at the start

    # Check that we have enough points pooled to carve up
    pool_all_labels_counts = []
    training_reqd_counts = []
    validation_reqd_counts = []
    testing_reqd_counts = []
    total_reqd_counts = []

    #    print('Total points collected from all files:')
    #    print('\tClassification Label\tTotal Points\t%\tAvg per File')
    #    print('\t--------------------\t------------\t-----\t------------')
    #    for label in range(total_labels):
    #        print('\t' + label_names[label] + ':\t\t' + str(label_counters[label]) +
    #              '\t\t' + str(np.round(100 * label_counters[label]/shape_tracker, decimals=2)) + '%' +
    #              '\t' + str(int(label_counters[label]/total_files_train))
    #              )

    for label in range(total_labels):

        pool_all_labels_counts.append(len(pool_all_labels_indices[label]))

        if label == total_labels - 1:

            # To account for any rounding errors when doing even splits, the last
            # label is supplemented by any otherwise lost points to ensure the
            # pool total is respected.
            training_reqd_counts.append(
                total_training_count - sum(training_reqd_counts[:total_labels])
            )
            validation_reqd_counts.append(
                total_validation_count
                - sum(validation_reqd_counts[:total_labels])
            )
            testing_reqd_counts.append(
                total_testing_count - sum(testing_reqd_counts[:total_labels])
            )

        else:

            this_label_training_ratio = training_label_fraction[label]
            this_label_validation_ratio = validation_label_fraction[label]
            this_label_testing_ratio = testing_label_fraction[label]

            training_reqd_counts.append(
                int(np.floor(total_training_count * this_label_training_ratio))
            )
            validation_reqd_counts.append(
                int(
                    np.floor(
                        total_validation_count * this_label_validation_ratio
                    )
                )
            )
            testing_reqd_counts.append(
                int(np.floor(total_testing_count * this_label_testing_ratio))
            )

        total_reqd_counts.append(
            training_reqd_counts[label]
            + testing_reqd_counts[label]
            + validation_reqd_counts[label]
        )

    print("\tPoints required for training, validation, and testing: ")
    print("\t--------------------\t------------\t------------\t------------")
    print("\tClassification Label\tRequired\tAvailable\tAcceptable?")
    print("\t--------------------\t------------\t------------\t------------")
    for label in range(total_labels):
        print(
            "\t"
            + label_names[label]
            + "\t\t"
            + str(total_reqd_counts[label])
            + "\t\t"
            + str(len(pool_all_labels_indices[label]))
            + "\t\t",
            end="",
            flush=True,
        )
        if len(pool_all_labels_indices[label]) < total_reqd_counts[label]:
            print(fn_etc.fancy("NO", "white", "red"))
        else:
            print(fn_etc.fancy("YES", "green", "black"))
    print("\t--------------------\t------------\t------------\t------------")
    total_reqd_points = sum(total_reqd_counts)
    total_avail_points = sum([len(x) for x in pool_all_labels_indices])
    print(
        "\tTotal\t\t\t"
        + str(total_reqd_points)
        + "\t\t"
        + str(total_avail_points)
        + "\t\t"
        + fn_etc.fancy("  NO  ", "white", "red")
        if total_avail_points < total_reqd_points
        else fn_etc.fancy("  OK  ", "white", "green")
    )

    if total_avail_points < total_reqd_points:
        ErrorMessage = (
            "Insufficient points!\r\n"
            + "Training, validation, and testing requires "
            + str(total_reqd_points)
            + " points but \r\n"
            + "we only have "
            + str(total_avail_points)
            + " points pooled from \r\n"
            + "all files in "
            + inputpath_train
            + ".\r\n"
            + "Check the table above for which type of points are mising and\r\n"
            + "either adjust the training requirements or supply additional training data."
        )
        fn_etc.err_msg(ErrorMessage)
        raise ValueError("Error, check the message above.")
    #    else:
    #        print('\x1b[36;1m\t' + 'OK!' + '\x1b[0m', flush=True)

    # ========================================================================
    #  Extract three sets of points of each label with
    #  one set each for training, validation, and testing sets
    # ========================================================================

    training_rnd_idx = np.empty((0,), dtype=int)
    validation_rnd_idx = np.empty((0,), dtype=int)
    testing_rnd_idx = np.empty((0,), dtype=int)

    for label in range(total_labels):
        print(
            "\tShuffling and drawing '"
            + label_names[label]
            + "' points for the training, validation, and testing pools...",
            end="",
        )
        training_startIdx = 0
        validation_startIdx = training_reqd_counts[label]
        testing_startIdx = (
            training_reqd_counts[label] + validation_reqd_counts[label]
        )

        training_rnd_idx = np.hstack(
            (
                training_rnd_idx,
                pool_all_labels_indices[label][
                    training_startIdx : training_reqd_counts[label]
                ],
            )
        )
        validation_rnd_idx = np.hstack(
            (
                validation_rnd_idx,
                pool_all_labels_indices[label][
                    validation_startIdx : validation_startIdx
                    + validation_reqd_counts[label]
                ],
            )
        )
        testing_rnd_idx = np.hstack(
            (
                testing_rnd_idx,
                pool_all_labels_indices[label][
                    testing_startIdx : testing_startIdx
                    + testing_reqd_counts[label]
                ],
            )
        )

    # mix up all the indicies from all the labels
    shuffle(training_rnd_idx)
    shuffle(validation_rnd_idx)
    shuffle(testing_rnd_idx)

    print("\x1b[36;1m\t" + "OK!" + "\x1b[0m", flush=True)

    # extract the input data and labels etc for all the indices
    X_training_distances_raw = points_pool_mmap[training_rnd_idx, :, :]
    Y_training_vectors_raw = vectors_pooled[training_rnd_idx]
    Y_training_binary_raw = binaries_pooled[training_rnd_idx]
    Y_training_labels_raw = labels_pooled[training_rnd_idx]

    X_validation_distances_raw = points_pool_mmap[validation_rnd_idx, :]
    Y_validation_vectors_raw = vectors_pooled[validation_rnd_idx]
    Y_validation_binary_raw = binaries_pooled[validation_rnd_idx]
    Y_validation_labels_raw = labels_pooled[validation_rnd_idx]

    X_testing_distances_raw = points_pool_mmap[testing_rnd_idx, :]
    Y_testing_vectors_raw = vectors_pooled[testing_rnd_idx]
    Y_testing_binary_raw = binaries_pooled[testing_rnd_idx]
    Y_testing_labels_raw = labels_pooled[testing_rnd_idx]

    # ========================================================================
    #  Save the Training, Validation, and Testing data
    # ========================================================================
    # TrainValTest_data_basename = time.strftime('%Y%m%d-%H%M') + '_nn' + str(ps_current['ClosestFriend']) + '-' + str(ps_current['FurthestFriend']) + '_Trn(' + str(training_type0_count) + 't0+' + str(training_type1_count) + 't1+' + str(training_type2_count) + 't2)_Val(' + str(validation_type0_count) + 't0+' + str(validation_type1_count) + 't1+' + str(validation_type2_count) + 't2))_Tes(' + str(testing_type0_count) + 't0+' + str(testing_type1_count) + 't1+' + str(testing_type2_count) + 't2).pkl'
    TrainValTest_data_basename = (
        time.strftime("%Y%m%d-%H%M")
        + "_nn"
        + str(ps_current["ClosestFriend"])
        + "-"
        + str(ps_current["FurthestFriend"])
        + "_"
        + str(total_labels)
        + "labels_Trn_Val_Tes.pkl"
    )
    TrainValTest_data_fname = os.path.join(
        s2_trnvaltest_outputpath, TrainValTest_data_basename
    )
    print(
        "\tSaving training, validation, and testing data to:\n\t\t"
        + TrainValTest_data_fname
        + " ... "
    )

    # copy the procsettings and remove entries that we don't need anymore
    ps = dict(ps_current)
    del ps["BinaryDumpShape"]
    del ps["DataDumpShape"]
    del ps["DistsDumpShape"]
    del ps["LabelsDumpShape"]
    del ps["FilePrefix"]
    del ps["datatable_mmap_fname"]
    del ps["dists_mmap_fname"]
    del ps["target_binary_mmap_fname"]
    del ps["target_vectors_mmap_fname"]
    del ps["target_labels_mmap_fname"]
    del ps["VectorsDumpShape"]

    # dump all our necessary arrays and shizz to a pickle. We first make a list of
    # all the variables that we are saving. This list is the first thing to go in
    # to the pickle and it's the first one we load. Doing this means we do not need
    # to keep track of the exact order in which everything was saved.
    save_order = [
        # training set
        "total_training_count",
        "training_label_fraction",
        "training_reqd_counts",
        "X_training_distances_raw",
        "Y_training_vectors_raw",
        "Y_training_binary_raw",
        "Y_training_labels_raw",
        # validation set
        "total_validation_count",
        "validation_label_fraction",
        "validation_reqd_counts",
        "X_validation_distances_raw",
        "Y_validation_vectors_raw",
        "Y_validation_binary_raw",
        "Y_validation_labels_raw",
        # testing set
        "total_testing_count",
        "testing_label_fraction",
        "testing_reqd_counts",
        "X_testing_distances_raw",
        "Y_testing_vectors_raw",
        "Y_testing_binary_raw",
        "Y_testing_labels_raw",
        # metadata
        "ps",
        "label_names",
    ]
    pickle_file = open(
        TrainValTest_data_fname, "wb"
    )  # open the file for writing
    pickle.dump(
        save_order, pickle_file, protocol=4
    )  # save the save_order first
    for item in save_order:
        pickle.dump(
            locals()[item], pickle_file, protocol=4
        )  # save each item in save_order to the pickle file
    pickle_file.close()  # close the pickle file

    # clean up the memmap data
    del points_pool_mmap
    os.remove(points_pool_fname)

    fn_etc.ok_msg(
        "Finished preparing training, validation, and testing datasets."
    )
    print("The input folder was\t" + inputpath_train)
    print("The output folder was\t" + s2_trnvaltest_outputpath)
    print(
        "-------------------------------------------------\n\tCompleted!\n-------------------------------------------------"
    )
    ### END preparing train-&-validation data pool.
