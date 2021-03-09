#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:28:41 2020

@author: andreas
"""
import glob
import subprocess
import os
from Basefolder import basefolder

filenamesList = glob.glob(basefolder+'Input/*.json')

print(filenamesList);
for i,fn in enumerate(filenamesList):
    fn_ = os.path.basename(fn);
    print('Running ',fn_,' ..');
    log = open(basefolder+'Output/'+fn_[:-5]+'.txt', 'a')
    command1 = subprocess.check_call(['./run.sh',basefolder,fn,str(i)],stdout=log, stderr=log)
    