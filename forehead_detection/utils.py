#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 22:09:47 2020
@author: chaitra
"""

import numpy as np

def connect_points(x0, y0, x1, y1, mat, inplace=False):
    x0=int(round(x0))
    y0=int(round(y0))
    x1=int(round(x1))
    y1=int(round(y1))

    if (x0, y0) == (x1, y1):
        mat[x0, y0] = 2
        return mat if not inplace else None
    # Swap axes if Y slope is smaller than X slope
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        mat = mat.T
        x0, y0, x1, y1 = y0, x0, y1, x1
    # Swap line direction to go left-to-right if necessary
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    # Write line ends
    mat[x0, y0] = 2
    mat[x1, y1] = 2
    # Compute intermediate coordinates using line equation
    x = np.arange(x0 + 1, x1)
    y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype)
    # Write intermediate coordinates
    mat[x, y] = 255
    if not inplace:
        return mat if not transpose else mat.T
    
def connect_keypoints(keypts, frame):
    
    #nose to left eye
    connect_points(keypts[0,0],keypts[0,1],keypts[1,0],keypts[1,1],frame)
    #nose to right eye
    connect_points(keypts[0,0],keypts[0,1],keypts[2,0],keypts[2,1],frame)
    #left eye to left ear
    connect_points(keypts[1,0],keypts[1,1],keypts[3,0],keypts[3,1],frame)
    #right eye to right ear
    connect_points(keypts[2,0],keypts[2,1],keypts[4,0],keypts[4,1],frame)
    #left shoulder to right shoulder
    connect_points(keypts[5,0],keypts[5,1],keypts[6,0],keypts[6,1],frame)
    #left shoulder to left elbow
    connect_points(keypts[5,0],keypts[5,1],keypts[7,0],keypts[7,1],frame)
    #left elbow to left wrist
    connect_points(keypts[7,0],keypts[7,1],keypts[9,0],keypts[9,1],frame)
    #right shoulder to right elbow
    connect_points(keypts[6,0],keypts[6,1],keypts[8,0],keypts[8,1],frame)
    #right elbow to right wrist
    connect_points(keypts[8,0],keypts[8,1],keypts[10,0],keypts[10,1],frame)
    #left shoulder to left hip
    connect_points(keypts[5,0],keypts[5,1],keypts[11,0],keypts[11,1],frame)
    #right shoulder to right hip
    connect_points(keypts[6,0],keypts[6,1],keypts[12,0],keypts[12,1],frame)
    #left hip to right hip
    connect_points(keypts[11,0],keypts[11,1],keypts[12,0],keypts[12,1],frame)
    #left hip to left knee
    connect_points(keypts[11,0],keypts[11,1],keypts[13,0],keypts[13,1],frame)
    #left knee to left ankle
    connect_points(keypts[13,0],keypts[13,1],keypts[15,0],keypts[15,1],frame)
    #right hip to right knee
    connect_points(keypts[12,0],keypts[12,1],keypts[14,0],keypts[14,1],frame)
    #right knee to right ankle
    connect_points(keypts[14,0],keypts[14,1],keypts[16,0],keypts[16,1],frame)
    
    return frame