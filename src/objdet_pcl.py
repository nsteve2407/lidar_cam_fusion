# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
from distutils.command.config import config
import cv2
import numpy as np
import torch

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
import zlib
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools
import open3d as o3d
from matplotlib import cm

import seaborn as sns
import matplotlib.pyplot as plt

# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")
    # print(pcl.shape)

    # step 1 : initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    # vis.get_render_option().background_color = np.asarray([0, 0, 0])
    # step 2 : create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])
    virdis = cm.get_cmap('viridis',8)
    intensity = pcl[:,3]
    intensity = np.where(intensity>0.5,0.5,intensity)
    intensity = np.where(intensity<0.0,0.0,intensity)
    pcd.colors = o3d.utility.Vector3dVector(virdis(intensity/0.3)[:,:3])

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    
    # view_ctl = vis.get_view_control()
    vis.add_geometry(pcd)
    vis.register_key_callback(262,callback_func=lambda v:v.close())
    vis.run()
    
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    
    vis.destroy_window()

    #######
    ####### ID_S1_EX2 END #######     
       

# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    for laser in frame.lasers:
        if laser.name==1:
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(zlib.decompress(laser.ri_return1.range_image_compressed))
            ri = np.array(ri.data).reshape(ri.shape.dims)
            break
    
    # step 2 : extract the range and the intensity channel from the range image
    r = ri[:,:,0]
    i = ri[:,:,1]
    # step 3 : set values <0 to zero
    r = np.where(r<0,0,r)
    i = np.where(i<0,0,i)

    max_r , max_i = np.max(r),np.max(i)
    max_i = 0.5
    i = np.where(i>max_i,max_i,i)
    
    
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    r = ((r/max_r)*255).astype(np.uint8)
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    i = ((i/max_i)*255).astype(np.uint8)
    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    
    img_range_intensity = np.vstack((r,i)) # remove after implementing all steps
    #######
    ####### ID_S1_EX1 END #######     
    # Crop to frontal fild of view
    
    return img_range_intensity[:,int(img_range_intensity.shape[1]/2 - img_range_intensity.shape[1]/4):int(img_range_intensity.shape[1]/2 + img_range_intensity.shape[1]/4)]


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######

    # print(configs)

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    unit = (configs.lim_x[1]-configs.lim_x[0])/configs.bev_height

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates   
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl[:,0] = lidar_pcl[:,0]/unit 

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    lidar_pcl[:,1] = (lidar_pcl[:,1]-configs.lim_y[0])/unit

    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    lidar_pcl_int = lidar_pcl.astype(np.int32)
    
    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######


    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    imap = np.zeros((configs.bev_height+1,configs.bev_width+1))

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    mask = np.lexsort((lidar_pcl[:,2]*-1,lidar_pcl_int[:,1],lidar_pcl_int[:,1]))
    lidar_pcl_int_z = lidar_pcl_int[mask]
    lidar_pcl_z = lidar_pcl[mask]
    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _,idx,counts = np.unique(lidar_pcl_int_z[:,:2],return_counts=True,return_index=True,axis=0)
    zvals = lidar_pcl_z[idx]
    # print('Lidar',lidar_pcl)
    # print('zvals',zvals)

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    # ids = np.lexsort((lidar_pcl[:,3]*-1,lidar_pcl_int[:,1],lidar_pcl_int[:,0]))
    # pts = lidar_pcl[ids]
    # pts[:,3] = np.where(pts[:,3]>0.5,0.5,pts[:,3])
    # pts[:,3] = np.where(pts[:,3]<0.0,0.0,pts[:,3])
    zvals[:,3] = np.where(zvals[:,3]>0.5,0.5,zvals[:,3])
    zvals[:,3] = np.where(zvals[:,3]<0.0,0.0,zvals[:,3])

    # _,ids = np.unique(pts[:,:2],return_index=True,axis=0)
    # intensity_vals = pts[ids]
    
    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    # for row in range(zvals.shape[0]):
    #     x,y,_,i = zvals[row,:]
    #     x,y = np.int32(x),np.int32(y)
    #     imap[x,y] = i/0.5
    imap[np.int_(zvals[:,0]),np.int_(zvals[:,1])] = zvals[:,3]/0.5
    # cv2.imshow('Intensity Image',imap)
    # cv2.waitKey(0)
    # plt.figure(figsize=(12,8))
    # # ax = sns.heatmap(imap,annot=True, fmt="d")
    # ax = plt.gca()
    # ax = sns.heatmap(imap)
    # plt.show()
    # #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    hmap = np.zeros((configs.bev_height+1,configs.bev_width+1))

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map


    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    # for row in range(zvals.shape[0]):
    #     x,y,z,_ = zvals[row,:]
    #     x,y = np.int32(x),np.int32(y)
    #     hmap[x,y] = z/(configs.lim_z[1]-configs.lim_z[0])
    hmap[np.int_(zvals[:,0]),np.int_(zvals[:,1])] = zvals[:,2]/(configs.lim_z[1]-configs.lim_z[0])

    # cv2.imshow('Height Image',hmap)
    # cv2.waitKey(0)
    #######
    ####### ID_S2_EX3 END #######       

    # TODO remove after implementing all of the above steps
    # lidar_pcl_cpy = []
    lidar_pcl_top = lidar_pcl_cpy
    height_map = hmap
    intensity_map = imap

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    # _, _, counts = np.unique(lidar_pcl_z[:, 0:2], axis=0, return_index=True, return_counts=True)
    # print(counts)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    # print('Norm COunts',np.unique(normalizedCounts))
    density_map[np.int_(zvals[:, 0]), np.int_(zvals[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # cv2.imshow('BEV Image',np.dstack((imap,hmap,density_map)))

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps


