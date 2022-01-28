#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates an isometric grid of the specified spacing

@author: dave

"""

import numpy as np

def iso_spots(boundary=[0,10000,0,10000], spacing=100):

    gridA_x = np.arange(boundary[0], boundary[1], spacing)
    gridA_y = np.arange(boundary[2], boundary[3], (np.sqrt(3)/2) * spacing)
    
    gridA_xv, gridA_yv = np.meshgrid(gridA_x, gridA_y, sparse=False, indexing='ij')
    gridA_yv[1::2] += 0.5 * spacing # or [::2] to modify the even indices

    grid_out = np.concatenate((gridA_xv.reshape(gridA_xv.size,1), gridA_yv.reshape(gridA_yv.size,1)), axis=1)
    
    # remove out-of-bounds xy pairs
    x_within = np.logical_and(grid_out[:,0] > boundary[0], grid_out[:,0] < boundary[1])
    y_within = np.logical_and(grid_out[:,1] > boundary[2], grid_out[:,1] < boundary[3])
    xy_within = np.where(np.logical_and(x_within, y_within))[0]
    
    grid_out_final = grid_out[xy_within, :]

    return grid_out_final

# Example usage:
#import matplotlib.pyplot as plt
#import os
#
#iso_grid = iso_spots(boundary=[0,40960,0,40960], spacing=50)
#print('Points: ' + str(iso_grid.shape[0]) + '\tDensity: ' + str(iso_grid.shape[0]/(40.96 * 40.96)))
#
#plt.scatter(iso_grid[:,0], iso_grid[:,1], marker='.', color='r', s=1)
#plt.gca().set_aspect('equal', adjustable='box')
#
## save grid
#iso_fname = os.path.join(os.getcwd(), 'iso_grid_3nmspacing.csv')
#np.savetxt(iso_fname, iso_grid, delimiter=',', header='x,y', comments='', fmt='%10.5f')

#start_xy = 0
#stop_xy = 100
#points_xy = 1000
#
#points_AB = int(points_xy / 2)
#points_AB_axis = int(np.sqrt(points_AB))
#
#inner_points = int((np.sqrt(points_AB) - 2)**2)