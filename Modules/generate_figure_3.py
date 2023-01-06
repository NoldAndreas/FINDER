#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pietro
"""
import glob
import os
import subprocess

from ComputeSeries import ComputeSeries

basefolder = os.path.dirname(os.getcwd())
datafolder = os.path.join(basefolder, "data_sources")
inputfolder = os.path.join(datafolder, "Input_fig3")

filenamesList = glob.glob(os.path.join(inputfolder, "*.json"))
filenamesList.sort()

outputfolder = os.path.join(datafolder, "Output")

if not os.path.isdir(outputfolder):
    os.mkdir(outputfolder)
    print("Creating the 'Ouput' directory in:")
    print(outputfolder)

print("Computing 'ComputeSeries' on the following files:")
print(filenamesList)
print()
#
for i, fn in enumerate(filenamesList):
    fn_ = os.path.basename(fn)
    print("Running ", fn_, " ..")
    file_name = os.path.join(inputfolder, fn_)
    ComputeSeries(datafolder, file_name, str(i))
