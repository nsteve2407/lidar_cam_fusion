# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############

        F = np.eye(6)
        F[0,3] = params.dt
        F[1,4] = params.dt
        F[2,5] = params.dt
        
        return F
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        q = params.q
        Q = np.zeros((6,6))
        Q[0,0] = (params.dt**3)*(q/3)
        Q[1,1] = (params.dt**3)*(q/3)
        Q[2,2] = (params.dt**3)*(q/3)
        Q[3,3] = params.dt*q
        Q[4,4] = params.dt*q
        Q[5,5] = params.dt*q
        Q[3,0] = (params.dt**2)*(q/2)
        Q[4,1] = (params.dt**2)*(q/2)
        Q[5,2] = (params.dt**2)*(q/2)
        Q[0,3] = (params.dt**2)*(q/2)
        Q[1,4] = (params.dt**2)*(q/2)
        Q[2,5] = (params.dt**2)*(q/2)
        return Q
        
    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        x_ = self.F()@track.x
        P_ = self.F()*track.P*self.F().transpose() + self.Q()

        track.set_x(x_)
        track.set_P(P_)

        pass
        

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        g = self.gamma(track,meas)
        S = self.S(track,meas,meas.sensor.get_H(track.x))
        K = track.P*meas.sensor.get_H(track.x).transpose()*np.linalg.inv(S)
        x = track.x + (K*g)
        P = (np.eye(track.P.shape[0])-K*meas.sensor.get_H(track.x))*track.P

        track.set_x(x)
        track.set_P(P)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        return meas.z - meas.sensor.get_hx(track.x)
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        return H*track.P*H.transpose() + meas.R
        
        ############
        # END student code
        ############ 