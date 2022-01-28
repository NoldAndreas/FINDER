#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 09:41:59 2018

@author: dave
"""

import numpy as np
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from matplotlib.path import Path


def drop_small_holes(shape, min_area):
    """
    Drop small inner-polygons from the main (outer) polygon.
    
    Any inner-polygon with an area smaller than min_area will be removed.
    i.e. larger holes will be preserved.
    """   
    assert shape.geom_type == 'Polygon'
    retained_inner_polygons = []
    for inner_polygon in shape.interiors:
        RawInnerPolyPts = np.array(inner_polygon.xy).T
        InteriorPts = MultiPoint(RawInnerPolyPts)
        InteriorPoly = Polygon(InteriorPts)
        if abs(InteriorPoly.area) >= min_area:
            retained_inner_polygons.append(inner_polygon)
    return Polygon(shape.exterior, retained_inner_polygons)


def cluster_shaper(cidx, Point_CluIDs, CluUIDs, xyData, NNDs, BufferEps, DoErodeBuffer, DropSmallHoles, demoted_points, demoted_clusters, reassigned_points, cluclub_out, ps):
    """
    cidx    Cluster Index (cidx=i, for i in range(0, total_clusters) when parallel processing)
    Point_CluIDs = novel_clusterclub
    CluUIDs	= UniqueClusIDs
    xyData	= datatable_mmapped
    """
    
    ThisClusterUID = CluUIDs[cidx]
    PtsIdx = np.where(Point_CluIDs == ThisClusterUID)[0]
    PtsThisCluster = np.shape(Point_CluIDs[PtsIdx])[0]
    
    MinPtsForACluster = 3
    
    VerboseOut = False
    
    if PtsThisCluster >= MinPtsForACluster: # we need 3 or more points to make a cluster

        demoted_flag = False

        # ========================================================================
        #  Expand member points into overlapping disc shapes
        # ========================================================================

        # Get x,y,UID for member points (points in this cluster)
        all_points = np.concatenate((xyData[PtsIdx, ps['xCol'], None],
                                     xyData[PtsIdx, ps['yCol'], None],
                                     xyData[PtsIdx, ps['UIDCol'], None]),
                                     axis=1)

        # determine the size of the buffer to convert points to an area
        if BufferEps == 'dims':
            buffer_estimate = (sum(all_points[:,0:2].max(axis=0) - all_points[:,0:2].min(axis=0)) / all_points.shape[0]) * 5
        elif BufferEps == 'dens':
            xyrange = all_points.max(axis=0) - all_points.min(axis=0)
            boundingarea = xyrange[0] * xyrange[1]
            if boundingarea == 0:
                # points in a line
                boundingarea = xyrange.max(axis=0) # If this is still zero then all the points have the same coordinates.
            buffer_estimate = np.sqrt((boundingarea / all_points.shape[0]) / np.pi)
        elif BufferEps == 'nnd':
            NNDs_this_cluster = NNDs[PtsIdx]
            buffer_estimate = 1.2 * np.mean(NNDs_this_cluster) # the nearest-neighbour distance for this point
        else:
            buffer_estimate = BufferEps

        # convert points to Shapely multipoint
        shpPoints = MultiPoint(np.array([all_points[:, 0], all_points[:, 1]]).T)

        # draw circle around each point and take the intersection of those shapes
        if DoErodeBuffer:
            # Expand the points and lines
            shpSpots = [p.buffer(2 * buffer_estimate) for p in shpPoints]
            shpPatches = unary_union(shpSpots).buffer(-buffer_estimate)
        else:
            shpSpots = [p.buffer(buffer_estimate) for p in shpPoints]
            shpPatches = unary_union(shpSpots)

        if shpPatches.area == 0:
            _, uniq_idx = np.unique(all_points, axis=0, return_index=True)
            raise ValueError('Cluster has an area of zero but contains ' + str(PtsThisCluster) + ' points. There are ' + str(all_points.shape[0] - uniq_idx.shape[0]) + ' duplicate xy points.')

        # ========================================================================
        # At this point we can be left with a polygon (which may contain holes) or 
        # a multipolygon e.g. because something got split off in the -buffer above
        # We can remove small extraneous polygons and holes and spin off larger
        # separated clusters as their own entities.
        # ========================================================================

        # ========================================================================
        #  remove small polygons and small holes
        # ========================================================================
        
        if shpPatches.type == "MultiPolygon":

            rebuild_shpPatches = []
            for polygon in shpPatches:
                poly_coords = np.array(polygon.exterior.xy).T
                p = Path(poly_coords) # make a matplotlib polygon to check how many points are inside it
                if sum(p.contains_points(shpPoints)) > 1:
                    # remove small holes from this polygon, if specified
                    if DropSmallHoles:
                        polygon = drop_small_holes(polygon, 0.5 * buffer_estimate)
                    rebuild_shpPatches.append(polygon)
                else: # only 1 point, demote it from clustering
                    demoted_ptID = all_points[p.contains_points(shpPoints), 2][0].astype(int)
                    if VerboseOut: print('Demoting isolated point ' + str(demoted_ptID)  + ' from cluster ' + str(ThisClusterUID))
                    demoted_points.append(demoted_ptID)
            
            if len(rebuild_shpPatches) == 1:
                shpPatches = Polygon(rebuild_shpPatches[0])
            elif len(rebuild_shpPatches) > 1:
                shpPatches = MultiPolygon(rebuild_shpPatches)
            else:
                if VerboseOut: print('Have managed to exclude all points in cluster; demoting ' + str(ThisClusterUID))
                demoted_clusters.append(ThisClusterUID)
                demoted_flag = True

        elif shpPatches.type == "Polygon":
            
            poly_coords = np.array(shpPatches.exterior.xy).T
            p = Path(poly_coords) # make a matplotlib polygon
        
            if sum(p.contains_points(shpPoints)) > 1:
                    # remove small holes, if required
                if DropSmallHoles:
                    shpPatches = drop_small_holes(shpPatches, 0.5 * buffer_estimate)
            else:
                demoted_clusters.append(ThisClusterUID)
                demoted_flag = True
                if VerboseOut: print('I don\'t think you should be able to get into this position regarding Cluster' + str(ThisClusterUID))

        # ========================================================================
        #  Multiple polygons - separate and assign new clusterIDs
        # ========================================================================

        if shpPatches.type == "MultiPolygon" and not demoted_flag:
            
            polygons = list(shpPatches)
            total_new_polygons = len(polygons)
            if VerboseOut: print('Splitting cluster ' + str(ThisClusterUID) + ' into ' + str(total_new_polygons) + ' new clusters')
            valid_sub_cluster = 0

            for polygon in polygons:
                                
                ShapeTmp = Path(np.array(polygon.exterior.xy).T)
                
                bool_contained_pts = ShapeTmp.contains_points(shpPoints)
                contained_ptIDs = all_points[bool_contained_pts, 2].astype(int)
                total_points_contained = sum(bool_contained_pts)
                
                #check we have enough enclosed points still to bother making this a new cluster
                if total_points_contained > MinPtsForACluster:
                    
                    valid_sub_cluster += 1

                    # record a NewCluUID built to avoid duplicating any existing ID.
                    maxCluUIDs = np.max(CluUIDs)                    # e.g. 61280
                    cluster_buffer1 = len(str(maxCluUIDs))          # e.g. 5
                    cluster_buffer2 = len(str(total_new_polygons)) + 1  # e.g. 3
                    
                    NewCluUID = int(str(maxCluUIDs) + str(ThisClusterUID).zfill(cluster_buffer1) + str(valid_sub_cluster).zfill(cluster_buffer2))

                    # record the pointIDs and their new clusterID
                    reassignment_cluID = np.array(NewCluUID).repeat(total_points_contained)
                    reassignment_tmp = np.hstack((contained_ptIDs[..., np.newaxis], reassignment_cluID[..., np.newaxis]))
                    reassigned_points.append(reassignment_tmp)

                    # build the output
                    my_cluclub_out = list()
                    my_cluclub_out.append(NewCluUID)
                    my_cluclub_out.append(ShapeTmp)
                    my_cluclub_out.append(polygon.area)
                    my_cluclub_out.append(total_points_contained)
                    my_cluclub_out.append(polygon.centroid.xy)
                
                    if VerboseOut: print('\tNew cluster ' + str(NewCluUID) + ' has ' + str(total_points_contained) + ' points')
                    
                    cluclub_out.append(my_cluclub_out) # export this cluster
                    
                else:
                    
                    if VerboseOut: print('\tNot enough points in split cluster. Demoting those points')
                    for demoted_pt in contained_ptIDs:
                        if VerboseOut: print('Demoting isolated point UID_' + str(demoted_pt) + ' from cluster ' + str(ThisClusterUID))
                        demoted_points.append(demoted_pt)

            # Check: we split up the multipolygon into new separated clusters and
            # kept track of how many had enough points to be considered as proper
            # clusters (in variable 'valid_sub_cluster'). If none of the new clusters made it
            # through (valid_sub_cluster == 0) then we demote this entire original cluster.
            if valid_sub_cluster == 0:                
                if VerboseOut: print('Splitting this cluster creates pathetic clusters. Demoting cluster ' + str(ThisClusterUID) + ' for insufficient points')
                demoted_clusters.append(ThisClusterUID)


        # ========================================================================
        #  Cluster is a single discrete polgon
        # ========================================================================
        
        elif shpPatches.type == "Polygon" and not demoted_flag:

            ClusterShape = Path(np.array(shpPatches.exterior.xy).T)

            points_contained = sum(ClusterShape.contains_points(shpPoints))

            # build the output
            my_cluclub_out = list()
            my_cluclub_out.append(ThisClusterUID)
            my_cluclub_out.append(ClusterShape)
            my_cluclub_out.append(shpPatches.area)
            my_cluclub_out.append(points_contained)
            my_cluclub_out.append(shpPatches.centroid.xy)
            
            cluclub_out.append(my_cluclub_out) # export this cluster

    else:

        if VerboseOut: print('Demoting cluster ' + str(ThisClusterUID) + ' for insufficient points')
        demoted_clusters.append(ThisClusterUID)

