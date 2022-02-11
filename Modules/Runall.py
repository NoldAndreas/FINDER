#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:28:41 2020

@author: andreas
"""
import glob
import subprocess
import os

from ComputeSeries import ComputeSeries


from Definitions import get_datafolder

basefolder = get_datafolder()+'/'

filenamesList = glob.glob(os.path.join(basefolder,'Input','*.json'))

if not os.path.isdir(basefolder+'Output/'):
    os.mkdir(basefolder+'Output/')
print("Computing 'ComputeSeries' on the following files:")
print(filenamesList)
print()

for i,fn in enumerate(filenamesList):
    fn_ = os.path.basename(fn)
    print('Running ',fn_,' ..')

    ComputeSeries(basefolder, basefolder+'Input/'+fn_, str(i))

    #log = open(basefolder+'Output/'+fn_[:-5]+'.txt', 'a')    
    #command1 = subprocess.check_call(['python','Modules/ComputeSeries.py',basefolder,fn,str(i)],stdout=log, stderr=log)
    #command1 = subprocess.check_call(f"python ComputeSeries.py {basefolder} {fn} {str(i)}",stdout=log, stderr=log)
    #command1 = subprocess.check_call(['./Modules/run.sh',basefolder,fn,str(i)],stdout=log, stderr=log)
    