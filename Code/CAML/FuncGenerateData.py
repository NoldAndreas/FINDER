#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions to support the creation of training and validation data.

A cell-synapse like outline to contain the points with the wider field of view.

@author: dave
"""

import numpy as np
import os
from shapely.geometry import Polygon, MultiPoint
from matplotlib.path import Path
from scipy.spatial import distance
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import matplotlib.patches as patches
from scipy import interpolate
import pickle

import pdb # debugging

def MakeCellOutline(ImageSize, Create_cSMAC, dname, fname, SavePreview):

    cell_verts = 50
    min_xy_dim = np.min((ImageSize[0],ImageSize[1]))
    cell_radius = min_xy_dim / 4
    cell_rad_var = cell_radius / 2.5
    cell_centreX = ImageSize[0] / 2
    cell_centreY = ImageSize[1] / 2

    InterpOversample = 20
    
    cell_vert_angle = np.linspace(0, 2 * np.pi, cell_verts, endpoint=False)
    cell_vert_radius = cell_radius + cell_rad_var * np.random.random(len(cell_vert_angle))

    cell_bounds_x = cell_centreX + cell_vert_radius * np.cos(cell_vert_angle)
    cell_bounds_y = cell_centreY + cell_vert_radius * np.sin(cell_vert_angle)

    # close the shape by copying the first five entries to the end
    cell_bounds_x = np.concatenate((cell_bounds_x, cell_bounds_x[0, None]), axis=0)
    cell_bounds_y = np.concatenate((cell_bounds_y, cell_bounds_y[0, None]), axis=0)
    
    # interpolate a smooth boundary
    TotalInterpPts = int(InterpOversample * cell_verts)
    tck, u = interpolate.splprep([cell_bounds_x, cell_bounds_y], s=0.0)
    x_interp, y_interp = interpolate.splev(np.linspace(0, 1, TotalInterpPts), tck)
    
    cell_bounds = np.concatenate((x_interp[:, None], y_interp[:, None]), axis=1)

    if Create_cSMAC:
        csmac_verts = 50
        csmac_radius = ImageSize[0] / 8
        csmac_rad_var = csmac_radius / 2
        
        csmac_vert_angle = np.linspace(0, 2 * np.pi, csmac_verts, endpoint=False)
        csmac_vert_radius = csmac_radius + csmac_rad_var * np.random.random(len(csmac_vert_angle))
        
        csmac_bounds_x = cell_centreX + csmac_vert_radius * np.cos(csmac_vert_angle)
        csmac_bounds_y = cell_centreY + csmac_vert_radius * np.sin(csmac_vert_angle)
        
        # close the shape by copying the first five entries to the end
        csmac_bounds_x = np.concatenate((csmac_bounds_x, csmac_bounds_x[0, None]), axis=0)
        csmac_bounds_y = np.concatenate((csmac_bounds_y, csmac_bounds_y[0, None]), axis=0)
        
        # interpolate a smooth boundary
        cSMACTotalInterpPts = int(InterpOversample * csmac_verts)
        csmac_tck, u = interpolate.splprep([csmac_bounds_x, csmac_bounds_y], s=0.0)
        x_csmac_interp, y_csmac_interp = interpolate.splev(np.linspace(0, 1, cSMACTotalInterpPts), csmac_tck)
        
        csmac_bounds = np.concatenate((x_csmac_interp[:, None], y_csmac_interp[:, None]), axis=1)
    else:
        csmac_bounds = np.empty((0,2))


    if SavePreview:
        plt.ioff()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, facecolor='none')
        plt.scatter(x_interp, y_interp, s=2, color='red')
        plt.scatter(cell_bounds_x, cell_bounds_y, s=5, color='blue')
        outline_patch = patches.Polygon(cell_bounds, facecolor='orange', edgecolor=None, alpha=0.5, lw=0.25, zorder=-2)
        ax.add_patch(outline_patch)
        
        if Create_cSMAC:
            plt.scatter(x_csmac_interp, y_csmac_interp, s=5, color='red')
            plt.scatter(csmac_bounds_x, csmac_bounds_y, s=5, color='blue')
            csmac_patch = patches.Polygon(csmac_bounds, facecolor='white', edgecolor=None, alpha=0.5, lw=0.25, zorder=-1)
            ax.add_patch(csmac_patch)
        
        ax.set_xlim(0, ImageSize[0])
        ax.set_ylim(0, ImageSize[1])
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        plt.tick_params(top=True, right=True, labeltop=False, labelright=False)
        #plt.box(False)
        ax.grid(clip_on=False, alpha=0.25)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_title(fname, size=5)
        plt.tight_layout()
        
        plt.savefig(os.path.join(dname, fname + ' - cell outline.png'),
                    dpi=300,
                    bbox_inches=0,
                    facecolor='none',
                    edgecolor='none',
                    transparent=True)
        plt.close()
        plt.ion()

    # save the outline data
    pickle.dump((cell_bounds, csmac_bounds), open(os.path.join(dname, fname + ' - cell outline.pkl'), 'wb' ), protocol=4)

    return(cell_bounds, csmac_bounds)


def find_dups(array1, array2):
    '''
    source: https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    '''
    tmp = np.prod(np.swapaxes(array1[:,:,None], 1, 2) == array2, axis = 2)
    return np.sum(np.cumsum(tmp, axis = 0) * tmp == 1, axis = 1).astype(bool)

def SprinklePoints(cluperum2,
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
                   DistributionType = 'Uniform',
                   UsecSMAC = False,
                   DepleteClustersFromcSMAC = True,
                   DepleteNCPointsFromcSMAC = False,
                   DepleteClustersFraction = 0.5,
                   DepleteNCPointsFraction = 0.75,
                   InputSeeds = None,
                   SeedOffset = None,
                   OffsetAssociationPercent = None
                   ):
   
    coord_res = 0.001
    doImages = False # This is mainly for debugging
    
    # even_nc_distribution = True # Distribute NC points evenly within the field?
    # Set to False to bias the distribution of NC points towards one side (i.e.
    # create an uneven 'background').
    
    if InputSeeds == None:
        UserSuppliedSeeds = False
    else:
        UserSuppliedSeeds = True
    
    cell_poly = Polygon(cell_bounds)
    cell_boundingbox = cell_poly.bounds
    noncluster_x_min = cell_boundingbox[0] / coord_res
    noncluster_x_max = cell_boundingbox[2] / coord_res
    noncluster_y_min = cell_boundingbox[1] / coord_res
    noncluster_y_max = cell_boundingbox[3] / coord_res

    noncluster_z_min = 0
    noncluster_z_max = ImageSize[2] / coord_res
    
    shrunk_cell_boundary = np.array(shrunk_cell_polygon.exterior.xy).T
    
    shrunk_cell_boundingbox = shrunk_cell_polygon.bounds
    cluster_x_min = shrunk_cell_boundingbox[0] / coord_res
    cluster_x_max = shrunk_cell_boundingbox[2] / coord_res
    cluster_y_min = shrunk_cell_boundingbox[1] / coord_res
    cluster_y_max = shrunk_cell_boundingbox[3] / coord_res
       
    if ImageSize[2] == 0:
        three_dee = False
        cluster_z_min = 0
        cluster_z_max = 0
    else:
        three_dee = True
        cluster_z_min = (0 + rad_max) / coord_res
        cluster_z_max = (ImageSize[2] - rad_max) / coord_res
    
    SimuCellArea = cell_poly.area
    
    cell_boundary = Path(cell_bounds)
    seed_boundary = Path(shrunk_cell_boundary)
    
    if UsecSMAC and csmac_bounds.shape[0] > 0:
        csmac_boundary = Path(csmac_bounds)
        csmac_poly = Polygon(csmac_bounds)
        cSMACArea = csmac_poly.area
    
    if doImages:
        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111, facecolor='none')
        plt.scatter(cell_bounds[:,0], cell_bounds[:,1], s=1, color='red')
        plt.scatter(shrunk_cell_boundary[:,0], shrunk_cell_boundary[:,1], s=2, color='grey')
        ax.set_xlim(0, ImageSize[0])
        ax.set_ylim(0, ImageSize[1])
        ax.set_aspect('equal')
        ax.set_title('Cluster Seeds')
        if UsecSMAC:
            plt.scatter(csmac_bounds[:,0], csmac_bounds[:,1], s=1, color='red')
    
    if ptsperclu > 0 and cluperum2 > 0 and percentptsinclu > 0:
       # clusters exist
    
        # Use or create cluster 'seeds' (centroid of cluster)
        if UserSuppliedSeeds:
            # TODO: this needs to be fixed as it won't work at present!
            Clusters_Required = InputSeeds.shape[0]

            Cluster_Seeds = np.zeros_like(InputSeeds)
            
#            # Our ClusterSeeds are loosely associated with our InputSeeds.
#            cell_vert_angle = 2 * np.pi * np.random.rand(InputSeeds.shape[0], 1)
#            Cluster_Seeds[:, 0] = InputSeeds[:, 0] + SeedOffset * np.cos(cell_vert_angle)
#            Cluster_Seeds[:, 1] = InputSeeds[:, 1] + SeedOffset * np.sin(cell_vert_angle)
#
#            # reallocate a portion of the supplied clusters to a random location
#            NonAssocClusterCount = np.floor((1 - (OffsetAssociationPercent / 100)) * Clusters_Required) # add this many seeds not associated to the input seeds
#            tmp_NonAssocSeeds = np.random.randint(0, high=np.floor(ImageSize / coord_res), size=(2 * NonAssocClusterCount, 2)) * coord_res # random xy coords
#            
#            NonAssocClusters_In_Cell = cell_poly.contains_points(MultiPoint(tmp_NonAssocSeeds)) # find the ones that are in the RoI boundary
#            tmp_NonAssocSeeds = tmp_NonAssocSeeds[NonAssocClusters_In_Cell,:]
#            Nonassocoffset = NonAssocClusterCount - tmp_NonAssocSeeds.shape[0] # work out if we are short on non-associated clusters
#
#            while Nonassocoffset != 0:
#                if Nonassocoffset > 0:
#                    # add more offset clusters
#                    tmp_NonAssocSeeds2 = np.random.rand(0, high=np.floor(ImageSize/coord_res), size=(Nonassocoffset,2)) * coord_res
#                    tmp_NonAssocSeeds = np.concatenate((tmp_NonAssocSeeds,tmp_NonAssocSeeds2), axis=0)
#                elif Nonassocoffset < 0:
#                    # delete clusters
#                    tmp_NonAssocSeeds[1:abs(Nonassocoffset),:].delete
#
#                NonAssocClusters_In_Cell = inpolygon(tmp_NonAssocSeeds[:, 0], tmp_NonAssocSeeds[:,1], cell_bounds[:,0], cell_bounds[:,1])
#                tmp_NonAssocSeeds = tmp_NonAssocSeeds[NonAssocClusters_In_Cell,:]
#                Nonassocoffset = NonAssocClusterCount - tmp_NonAssocSeeds.shape[0]
#
#            Cluster_Seeds[1:NonAssocClusterCount,:] = tmp_NonAssocSeeds # replace the existing supplied seeds with the random ones.
            
        else:
            # Create fresh cluster seeds
            if UsecSMAC:
                Clusters_Required = int(np.floor(((SimuCellArea - cSMACArea)/1E6) * cluperum2))
            else:
                Clusters_Required = int(np.floor((SimuCellArea/1E6) * cluperum2))
            
            SeedBank = np.zeros((Clusters_Required,3))
            
            GoodFirstSeed = False
            while GoodFirstSeed == False:
                FirstClusterSeed_x = np.random.randint(cluster_x_min, high=cluster_x_max, size=1) * coord_res
                FirstClusterSeed_y = np.random.randint(cluster_y_min, high=cluster_y_max, size=1) * coord_res
                if three_dee:
                    FirstClusterSeed_z = np.random.randint(cluster_z_min, high=cluster_z_max, size=1) * coord_res
                else:
                    FirstClusterSeed_z = np.array([cluster_z_max])
                FirstClusterSeed = np.concatenate((FirstClusterSeed_x[None, :], FirstClusterSeed_y[None, :], FirstClusterSeed_z[None, :]), axis=1)
                GoodFirstSeed = seed_boundary.contains_points(FirstClusterSeed[:,:2])[0]
                if GoodFirstSeed and UsecSMAC:
                    FirstIncSMAC = csmac_boundary.contains_points(FirstClusterSeed[:,:2])[0]
                    DiceRoll = np.random.rand(1)[0]
                    if DepleteClustersFromcSMAC and FirstIncSMAC and DiceRoll < DepleteClustersFraction:
                        GoodFirstSeed = False

            SeedBank[0,:] = FirstClusterSeed[None, :]
            
            GoodSeeds = 1
            
            while GoodSeeds < Clusters_Required:
                ProbationaryCluster = True
                ProposedSeedID = GoodSeeds + 1

                AllocationAttempt = 0
                
                while ProbationaryCluster and AllocationAttempt < 10:
                    # generate a new cluster seed
                    ProposedClusterSeed_x = np.random.randint(cluster_x_min, high=cluster_x_max, size=1) * coord_res
                    ProposedClusterSeed_y = np.random.randint(cluster_y_min, high=cluster_y_max, size=1) * coord_res
                    if three_dee:
                        ProposedClusterSeed_z = np.random.randint(cluster_z_min, high=cluster_z_max, size=1) * coord_res
                    else:
                        ProposedClusterSeed_z = np.array([cluster_z_max])
                    ProposedClusterSeed = np.concatenate((ProposedClusterSeed_x[None, :], ProposedClusterSeed_y[None, :], ProposedClusterSeed_z[None, :]), axis=1)
                    
                    # check if it has fallen inside the cell shape
                    ProposedInside = seed_boundary.contains_points(ProposedClusterSeed[:,:2])[0]
                    
                    # check if we are in the cSMAC
                    if ProposedInside and UsecSMAC:
                        ProposedIncSMAC = csmac_boundary.contains_points(ProposedClusterSeed[:,:2])[0]
                        DiceRoll = np.random.rand(1)[0]
                        if DepleteClustersFromcSMAC and ProposedIncSMAC and DiceRoll < DepleteClustersFraction:
                            ProposedInside = False                        
                    
                    if ProposedInside: # check we are not too close to any other seed
                        # get distances to other seeds
                        otherseeds = SeedBank[:GoodSeeds,:]
                        seed_dists = distance.cdist(ProposedClusterSeed,otherseeds)

                        if np.min(seed_dists) > 2.2 * rad_max:       # seed must be at least 4xRadius away plus a small buffer to force clear segregation of finished clusters
                            SeedBank[ProposedSeedID-1, :] = ProposedClusterSeed
                            ProbationaryCluster = False
                            GoodSeeds += 1
                        else:
                            AllocationAttempt += 1
               
                if AllocationAttempt >= 10:
                    Clusters_Required -= 1
                    #print('Abandoning this cluster attempt! (' + str(ProposedSeedID) + ' of ' + str(Clusters_Required) + ')')
                
            Cluster_Seeds = SeedBank[:Clusters_Required, :] # copy the valid seeds
            
            # convert our cluster seeds into a collection of patches
            # we can use this to exclude NC points from falling too close to a 
            # cluster seed where they might conflict with clustered points
            Cluster_MultiPts = MultiPoint(Cluster_Seeds)
            Cluster_Spots = [p.buffer(rad_max) for p in Cluster_MultiPts]
            Cluster_Patches = unary_union(Cluster_Spots)
            
        # End of cluster seed import or generation
        
        if doImages:
            plt.scatter(Cluster_Seeds[:,0], Cluster_Seeds[:,1], s=10, color='black', marker='+')
            
            for patch_idx, patch_polygon in enumerate(Cluster_Patches):
                patch_p = Path(np.array(patch_polygon.exterior.xy).T)
                patch = patches.PathPatch(patch_p, facecolor='yellow', edgecolor=None, alpha=0.5, lw=0.25, zorder=-1)
                ax.add_patch(patch)

        # Calculate the distribution of events in and out of clusters
        TotalInClusterPoints = int(np.ceil(ptsperclu * Clusters_Required)) # our in-cluster points are clusters * points-per-cluster.
        TotalEvents = int(np.ceil((TotalInClusterPoints / (percentptsinclu / 100)))) # if x percent of points are in-clusters then 1/x points are there in total.
        TotalExClusterPoints = TotalEvents - TotalInClusterPoints # our ex-cluster points are therefore the difference.

        NCPointsList = np.zeros((1,3)) #init using empty gives bizarre results so we init the array with a single entry and delete it later

        if TotalExClusterPoints > 0:
                    
            NCPointsRemaining = TotalExClusterPoints # this is how many points we are behind in our goal
            OvercompensationFactor = 2 # over-estimate so that we don't go through this process too many times

            while NCPointsRemaining != 0:

                if NCPointsRemaining > 0:
                    # we need to generate more points
                    Generating_NCPoints = int(NCPointsRemaining * OvercompensationFactor) 

                    NCPointsList_x = np.random.randint(noncluster_x_min, high=noncluster_x_max, size=(Generating_NCPoints,1)) * coord_res
                    NCPointsList_y = np.random.randint(noncluster_y_min, high=noncluster_y_max, size=(Generating_NCPoints,1)) * coord_res
                    if three_dee:
                        NCPointsList_z = np.random.randint(noncluster_z_min, high=noncluster_z_max, size=(Generating_NCPoints,1)) * coord_res
                    else:
                        NCPointsList_z = np.full((Generating_NCPoints,1), noncluster_z_max)
                    NCPtsProposed = np.concatenate((NCPointsList_x, NCPointsList_y, NCPointsList_z), axis=1)
                    
                    # Find events within the cell RoI
                    NCPoints_In_Cell = cell_boundary.contains_points(NCPtsProposed[:,:2])
                    NCPtsProposed = NCPtsProposed[NCPoints_In_Cell, :]

                    # remove points if they are already in NCPointsList
                    # unique_NCpts = np.logical_not(find_dups(NCPtsProposed, NCPointsList))
                    unique_NCpts = np.logical_not(np.isin(NCPtsProposed, NCPointsList).all(axis=1))
                    NCPtsProposed = NCPtsProposed[unique_NCpts, :]
                    if np.sum(unique_NCpts) < NCPtsProposed.shape[0]:
                        print('\n\t***\t' + str(NCPointsList.shape[0] - unique_NCpts.shape[0]) + ' Proposed NC points with duplicate xy coordinates\n')

                    # deplete points in the SMAC regions, as required
                    if UsecSMAC and NCPtsProposed.shape[0] > 0:
                        DiceRoll = np.random.rand(NCPtsProposed.shape[0], 1) > DepleteNCPointsFraction
                        if DepleteNCPointsFromcSMAC:
                            KeepNCPoints_SafeZone = np.invert(csmac_boundary.contains_points(NCPtsProposed[:,:2])) # True for points outside cSMAC
                        else:
                            KeepNCPoints_SafeZone = csmac_boundary.contains_points(NCPtsProposed[:,:2]) # True for points inside the cSMAC
                        KeepNCPoints = np.array([ x|y|z[0] for (x,y,z) in zip(KeepNCPoints_SafeZone, DiceRoll)]) # True for points in the safe zone OR points surviving dice-roll
                        
                        NCPtsProposed = NCPtsProposed[KeepNCPoints, :]
                    
                    # if we still have points to play with, remove those which fell inside the cluster 'zones'
                    if NCPtsProposed.shape[0] > 0:
                        NCPointsInAnyCluster = np.zeros((NCPtsProposed.shape[0], Cluster_Seeds.shape[0]), dtype=bool)
                        for pidx, polygon in enumerate(Cluster_Patches):
                            poly_coords = np.array(polygon.exterior.xy).T
                            p = Path(poly_coords)
                            NCPointsInAnyCluster[:, pidx] = p.contains_points(NCPtsProposed[:,:2])
    
                        KeptPoints = np.invert(np.any(NCPointsInAnyCluster, axis=1))
                        NCPtsProposed = NCPtsProposed[KeptPoints,:]
                        NCPointsList = np.concatenate((NCPointsList, NCPtsProposed))

                        # remove duplicate xy points in NCPointsList
                        _, uniq_idx = np.unique(NCPointsList, axis=0, return_index=True)
                        if uniq_idx.shape[0] < NCPointsList.shape[0]:
                            NCPointsList = NCPointsList[uniq_idx,:]
                            print('\n\t***\t3 - ' + str(NCPointsList.shape[0] - uniq_idx.shape[0]) + ' NC points with duplicate xy coordinates\n')

                        NCPointsRemaining = int(TotalExClusterPoints - NCPointsList.shape[0]) # re-test for sufficient number of events
                        
                    else:
                        print('managed to remove all generated points. this may become a problem!')

                    NCPtsProposed = None
                    OvercompensationFactor = 2 * OvercompensationFactor # bump up the overcompensation for the next round lest we get stuck in a situation generating only a few points which don't land where we want them
                    
                    if OvercompensationFactor > 100:
                        raise ValueError('Taking to long to position NC Points in this clustered image. Please check your clustering parameters.')
                        
                elif NCPointsRemaining < 0:
                    # we have too many events, remove them

                    # remove duplicate xy points in NCPointsList
                    _, uniq_idx = np.unique(NCPointsList, axis=0, return_index=True)
                    if uniq_idx.shape[0] < NCPointsList.shape[0]:
                        NCPointsList = NCPointsList[uniq_idx,:]
                        print('\n\t***\t4A - ' + str(NCPointsList.shape[0] - uniq_idx.shape[0]) + ' NC points with duplicate xy coordinates\n')

                    # random permutation of the NCPointsList indices, minus 1 to take account of the init row, then plus one to avoid index=0
                    RandomKeeperIdx = np.random.permutation(NCPointsList.shape[0] - 1) + 1
                    RandomKeeperIdx = np.concatenate(([0], np.sort(RandomKeeperIdx[:TotalExClusterPoints])))
                    NCPointsList = NCPointsList[RandomKeeperIdx,:] 
                    
                    # remove duplicate xy points in NCPointsList
                    _, uniq_idx = np.unique(NCPointsList, axis=0, return_index=True)
                    if uniq_idx.shape[0] < NCPointsList.shape[0]:
                        NCPointsList = NCPointsList[uniq_idx,:]
                        print('\n\t***\t4B - ' + str(NCPointsList.shape[0] - uniq_idx.shape[0]) + ' NC points with duplicate xy coordinates\n')

                    NCPointsRemaining = int(TotalExClusterPoints - RandomKeeperIdx.shape[0] + 1) # re-test for sufficient number of events to exit the while-loop

        NCPointsList = NCPointsList[1:,:] # delete the initialization row
        # Add the cluster UID column (which cluster do they belong to?) and cluster Label column (which type of clustering do we have?)
        # for NC points they belong to no cluster (UID=0) and they are Type 0
        NCPointsList = np.concatenate((NCPointsList, np.zeros((NCPointsList.shape[0], 2))), axis=1)

        if doImages:
            plt.scatter(NCPointsList[:,0], NCPointsList[:,1], s=2, color='cyan')
        
        # Add events around the cluster seeds
        ClusteredPointsList = np.zeros((TotalInClusterPoints, 5)) #init an empty array: [ x, y, z, clusID, Label ] We will ignore this first row later.
        InsertionPoint = 0
        for clusterID in range(Clusters_Required):

            tmp_pts_to_generate = ptsperclu
            ClusteredPointsList_final = np.zeros((1, 5)) # x, y, z, cluster UID, Label
            
            while tmp_pts_to_generate > 0:
                
                ClusteredPointsList_tmp = np.zeros((tmp_pts_to_generate, 5)) # x, y, z, cluster UID, Label
                
                if DistributionType == 'Uniform':
                
                    Edge_fuzzing = Edge_fuzz_ratio * (rad_max - rad_min)
                
                    # uniform distribution
                    min_dist = int(np.floor(((rad_min + Edge_fuzzing)**2 / coord_res)))
                    max_dist = int(np.floor(((rad_max - Edge_fuzzing)**2 / coord_res)))
                    CluPtAngles_phi = 2 * np.pi * np.random.rand(tmp_pts_to_generate, 1) # phi
                    CluPtAngles_theta = np.arccos(2 * np.random.rand(tmp_pts_to_generate, 1) - 1) # theta
                    CluPtRadii = np.sqrt(np.random.randint(min_dist, high=max_dist, size=(tmp_pts_to_generate,1)) * coord_res) # radii, range rmin to rmax
    
                    ClusteredPointsList_tmp[:,0, None] = np.array((CluPtRadii * np.cos(CluPtAngles_phi) * np.sin(CluPtAngles_theta)) + Cluster_Seeds[clusterID, 0])
                    ClusteredPointsList_tmp[:,1, None] = np.array((CluPtRadii * np.sin(CluPtAngles_phi) * np.sin(CluPtAngles_theta)) + Cluster_Seeds[clusterID, 1])
                    ClusteredPointsList_tmp[:,2, None] = np.array((CluPtRadii * np.cos(CluPtAngles_theta)) + Cluster_Seeds[clusterID, 2])

                    if Edge_fuzz_ratio > 0:
                        # fuzzle the points to blur out hard-edge clusters
                        fuzzy_x_offsets = np.random.normal(0, Edge_fuzzing, tmp_pts_to_generate)
                        fuzzy_y_offsets = np.random.normal(0, Edge_fuzzing, tmp_pts_to_generate)
                        fuzzy_z_offsets = np.random.normal(0, Edge_fuzzing, tmp_pts_to_generate)
                        ClusteredPointsList_tmp[:,0, None] = ClusteredPointsList_tmp[:,0, None] + fuzzy_x_offsets[:, None]
                        ClusteredPointsList_tmp[:,1, None] = ClusteredPointsList_tmp[:,1, None] + fuzzy_y_offsets[:, None]
                        ClusteredPointsList_tmp[:,2, None] = ClusteredPointsList_tmp[:,2, None] + fuzzy_z_offsets[:, None]

                elif DistributionType == 'Gaussian':

                    # Gaussian distribution start
                    mean = np.array([Cluster_Seeds[clusterID, 0], Cluster_Seeds[clusterID, 1], Cluster_Seeds[clusterID, 2]])
                    sigma = np.array([0.5*rad_max, 0.5*rad_max, 0.5*rad_max]) # decrease/increase the multiplier to generate tighter/looser clustering about the cluster seed
                    covariance = np.diag(sigma ** 2)
                    xyz_gaussian = np.random.multivariate_normal(mean, covariance, ptsperclu)
                    ClusteredPointsList_tmp[:,0, None] = xyz_gaussian[:,0,None]
                    ClusteredPointsList_tmp[:,1, None] = xyz_gaussian[:,1,None]
                    ClusteredPointsList_tmp[:,2, None] = xyz_gaussian[:,2,None]

                ClusteredPointsList_tmp[:,3] = clusterID + 1
                ClusteredPointsList_tmp[:,4] = 1 # the label for clustered points
    
                if Uncertainty_fwhm > 0:
                    # add localization uncertainty to each point
                    locprec_x_offsets = np.random.normal(0, Uncertainty_fwhm, tmp_pts_to_generate)
                    locprec_y_offsets = np.random.normal(0, Uncertainty_fwhm, tmp_pts_to_generate)
                    locprec_z_offsets = np.random.normal(0, Uncertainty_fwhm, tmp_pts_to_generate)
                    # Todo: combine and save these uncertainty values for each point
                    ClusteredPointsList_tmp[:,0, None] = ClusteredPointsList_tmp[:,0, None] + locprec_x_offsets[:, None]
                    ClusteredPointsList_tmp[:,1, None] = ClusteredPointsList_tmp[:,1, None] + locprec_y_offsets[:, None]
                    ClusteredPointsList_tmp[:,2, None] = ClusteredPointsList_tmp[:,2, None] + locprec_z_offsets[:, None]

                # remove duplicate xy points in ClusteredPointsList_tmp
                _, uniq_idx = np.unique(ClusteredPointsList_tmp, axis=0, return_index=True)
                if uniq_idx.shape[0] < ClusteredPointsList_tmp.shape[0]:
                    uniq_idx = np.sort(uniq_idx)
                    ClusteredPointsList_tmp = ClusteredPointsList_tmp[uniq_idx,:]
                
                # remove points if they are already in ClusteredPointsList_final
                # duplicate_filter = np.logical_not(find_dups(ClusteredPointsList_tmp, ClusteredPointsList_final))
                unique_CPts = np.logical_not(np.isin(ClusteredPointsList_tmp, ClusteredPointsList_final).all(axis=1))
                ClusteredPointsList_tmp = ClusteredPointsList_tmp[unique_CPts,:]
                    
                ClusteredPointsList_final = np.concatenate((ClusteredPointsList_final, ClusteredPointsList_tmp[uniq_idx,:]), axis=0)
                tmp_pts_to_generate -= ClusteredPointsList_tmp.shape[0]

            ClusteredPointsList[InsertionPoint:InsertionPoint+ptsperclu, :] =  ClusteredPointsList_final[1:,:] # ignoring the first row which is all zeros.
            InsertionPoint += ptsperclu

        if doImages:
            plt.scatter(ClusteredPointsList[:,0], ClusteredPointsList[:,1], s=2, color='red')
            
        # combine the clustered and non-clustered points
        PointList = np.concatenate((ClusteredPointsList, NCPointsList), axis=0)
        
        # Check for duplicate points
        if ClusteredPointsList.shape[0] > 0:
            _, uniq_idx = np.unique(ClusteredPointsList[:,:2], axis=0, return_index=True)
            if uniq_idx.shape[0] < ClusteredPointsList.shape[0]:
                print('\t***\t' + str(ClusteredPointsList.shape[0] - uniq_idx.shape[0]) + ' Clustered points with duplicate xy coordinates')
        
        if NCPointsList.shape[0] > 0:
            _, uniq_idx = np.unique(NCPointsList[:,:2], axis=0, return_index=True)
            if uniq_idx.shape[0] < NCPointsList.shape[0]:
                print('\t***\t' + str(NCPointsList.shape[0] - uniq_idx.shape[0]) + ' NC points with duplicate xy coordinates')
        
        if PointList.shape[0] > 0:
            _, uniq_idx = np.unique(PointList[:,:2], axis=0, return_index=True)
            if uniq_idx.shape[0] < PointList.shape[0]:
                print('\t***\t' + str(PointList.shape[0] - uniq_idx.shape[0]) + ' overall points with duplicate xy coordinates')
        else:
            raise ValueError('Somehow failed to get this far without generating any points!')

        
    else:
        # There are no clusters to make, i.e. CSR
        # if we have zero for any of clusters/um or events/cluster or
        # percent events in clusters then we can't have any clusters.
        
        Cluster_Seeds = np.zeros((1,3)) # function will return an empty list of seeds
        
        TotalNonClusterPoints = int(np.floor(ptsperum2 * (SimuCellArea / 1E6)))
      
        PointList = np.zeros((1,3)) #init array
        
        NCPointsRemaining = TotalNonClusterPoints # this is how many points we are behind in our goal
        OvercompensationFactor = 2 # over-estimate so that we don't go through this process too many times

        while NCPointsRemaining != 0:
            if NCPointsRemaining > 0:
                # We need to generate points

                Generating_NCPoints = int(NCPointsRemaining * OvercompensationFactor) # over-estimate so that we don't go through this process too many times

                NCPointsList_x = np.random.randint(noncluster_x_min, high=noncluster_x_max, size=(Generating_NCPoints,1)) * coord_res
                NCPointsList_y = np.random.randint(noncluster_y_min, high=noncluster_y_max, size=(Generating_NCPoints,1)) * coord_res
                if three_dee:
                    NCPointsList_z = np.random.randint(noncluster_z_min, high=noncluster_z_max, size=(Generating_NCPoints,1)) * coord_res
                else:
                    NCPointsList_z = np.full((Generating_NCPoints,1), noncluster_z_max)
                NCPtsProposed = np.concatenate((NCPointsList_x, NCPointsList_y, NCPointsList_z), axis=1)
                
                # Find events within the cell RoI
                NCPoints_In_Cell = cell_boundary.contains_points(NCPtsProposed[:,:2])
                NCPtsProposed = NCPtsProposed[NCPoints_In_Cell,:]
                
                if NCPtsProposed.shape[0] > 0:
                
                    # check for duplicate points and remove them
                    _, uniq_idx = np.unique(NCPtsProposed, axis=0, return_index=True)
                    if uniq_idx.shape[0] < NCPtsProposed.shape[0]:
                        uniq_idx = np.sort(uniq_idx)
                        NCPtsProposed = NCPtsProposed[uniq_idx,:]
    
                    if UsecSMAC and NCPtsProposed.shape[0] > 0:
                        DiceRoll = np.random.rand(NCPtsProposed.shape[0], 1) > DepleteNCPointsFraction
                        if DepleteNCPointsFromcSMAC:
                            KeepNCPoints_SafeZone = np.invert(csmac_boundary.contains_points(NCPtsProposed[:,:2])) # True for points outside cSMAC
                        else:
                            KeepNCPoints_SafeZone = csmac_boundary.contains_points(NCPtsProposed[:,:2]) # True for points inside the cSMAC
                        KeepNCPoints = np.array([ x|y|z[0] for (x,y,z) in zip(KeepNCPoints_SafeZone, DiceRoll)]) # True for points in the safe zone OR points surviving dice-roll
                        if KeepNCPoints.shape[0] == 0:
                            pdb.set_trace()
                        NCPtsProposed = NCPtsProposed[KeepNCPoints, :]
                    
                    PointList = np.concatenate((PointList, NCPtsProposed))

                NCPointsRemaining = int(TotalNonClusterPoints - PointList.shape[0]) # re-test for sufficient number of events
                
                NCPtsProposed = None
                OvercompensationFactor = OvercompensationFactor * 2
                
                if OvercompensationFactor > 100:
                        raise ValueError('Taking to long to position NC Points in this clustered image. Please check your clustering parameters.')
            
            elif NCPointsRemaining < 0:
                # we have too many points, remove them
                # random permutation of the PointList indices, minus 1 to take account of the init row, then plus one to avoid index=0
                RandomKeeperIdx = np.random.permutation(PointList.shape[0] - 1) + 1
                RandomKeeperIdx = np.concatenate(([0], np.sort(RandomKeeperIdx[:TotalNonClusterPoints - 1])))
                PointList = PointList[RandomKeeperIdx,:] 
                NCPointsRemaining = int(TotalNonClusterPoints - PointList.shape[0]) # re-test for sufficient number of events to exit the while-loop

        PointList = PointList[1:,:] # delete the initialization row
        PointList = np.concatenate((PointList, np.zeros((PointList.shape[0], 2))), axis=1) # Add clusterUID and Label values (both are 0 for NC points)

        if doImages:
            plt.scatter(PointList[:,0], PointList[:,1], s=2, color='cyan')
            
        if PointList.shape[0] > 0:
            _, uniq_idx = np.unique(PointList[:,:2], axis=0, return_index=True)
            if uniq_idx.shape[0] < PointList.shape[0]:
                print('\t***\t' + str(PointList.shape[0] - uniq_idx.shape[0]) + ' CSR points with duplicate xy coordinates')
        else:
            raise ValueError('Somehow failed to get this far without generating any points!')

    return(PointList, Cluster_Seeds)
