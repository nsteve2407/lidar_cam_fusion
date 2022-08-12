# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 
from student.filter import Filter
class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        self.KF = Filter()
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        self.unassigned_tracks = []
        self.unassigned_meas = []
        # the following only works for at most one track and one measurement
        N,M = len(track_list),len(meas_list)
        self.association_matrix = np.matrix([]) # reset matrix
         # reset lists
        
        if len(meas_list) > 0:
            self.unassigned_meas = list(range(M))
        if len(track_list) > 0:
            self.unassigned_tracks = list(range(N))
        if len(meas_list) > 0 and len(track_list) > 0: 
            self.association_matrix = np.ones((N,M))*np.inf

            for i in range(N):
                t = track_list[i]
                for j in range(M):
                    m = meas_list[j]
                    mh_dist = self.MHD(t,m,self.KF)
                    if (self.gating(mh_dist,m.sensor)):
                        self.association_matrix[i,j] = mh_dist


        
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        min_val = np.min(self.association_matrix)
        if min_val<np.inf:        
            # remove from list
            i,j = np.unravel_index(np.argmin(self.association_matrix),self.association_matrix.shape)
            self.association_matrix[i,:] = np.inf
            self.association_matrix[:,j] = np.inf
            self.unassigned_tracks.remove(i) 
            self.unassigned_meas.remove(j)
            
            return i,j
                
        ############
        # END student code
        ############ 
        return np.nan, np.nan 

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        
        if sensor.name =='lidar':
            if MHD<= params.gating_val_lidar: return True
        else:
            if MHD<= params.gating_val_camera: return True

        return False
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas,KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        
        g = KF.gamma(track,meas)
        H = meas.sensor.get_H(track.x)
        S = KF.S(track,meas,H)
        
        return g.transpose()@np.linalg.inv(S)@g
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)