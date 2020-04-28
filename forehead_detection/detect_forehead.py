import numpy as np
from PIL import Image
from pose_engine import PoseEngine
import json
from copy import deepcopy
import matplotlib.pyplot as plt
import cv2
import time

def forehead_coords(engine, img, thermal_shape, h, v, im_x=641, im_y=481):
    
    # Posenet getting keypoints
    pil_image = img
    im_x, im_y = img.size
    engine = PoseEngine('./forehead_detection/models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
    poses, keypoints, img, inference_time = engine.DetectPosesInImage(np.uint8(pil_image),0)
    # Getting forehead coordinates for 10 people in frame (NOTE: posenet currently processes a maximum of 10 people in the frame)
    keypoints_eyes = (keypoints[:,1,:], keypoints[:,2,:])
    lips_keypoints = []
    forehead_keypoints = []
    transformed_keypoints = []
    x_ratio = (im_x*h)/thermal_shape[0] 
    y_ratio = (im_y*v)/thermal_shape[1]
    x_bias = (im_x*h-im_x)/2
    y_bias = (im_y*v-im_y)/2
    for i in range(10):
        dist = abs(keypoints_eyes[0][i][1]-keypoints_eyes[1][i][1])
        # Format: ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y+offset)) offset is the distance wanted above the eyes.
        forehead_keypoints.append([(int(keypoints_eyes[0][i][1]+dist*0.5),int(keypoints_eyes[0][i][0]+dist*0.2)), (int(keypoints_eyes[1][i][1]-dist*0.5), int(keypoints_eyes[1][i][0]-dist*1.5))])     
        # Transform coordinates
        if keypoints_eyes[0][i][1]==0:
            xL, yL, xR, yR = 0,0,0,0
        else:
            # include some hard-coded emperical fix
            e = np.e
            xL = (forehead_keypoints[i][0][0]+ x_bias) /x_ratio 
            yL = (forehead_keypoints[i][0][1]+ y_bias) /y_ratio 
            xR = (forehead_keypoints[i][1][0] + x_bias) /x_ratio
            yR = (forehead_keypoints[i][1][1] + y_bias) /y_ratio
            size_factor = abs(xL-xR)/thermal_shape[0]
            size_factor=0.08*e**size_factor
            v_dis = abs((xR+xL)/2-thermal_shape[0])
            ori_factor = abs(xR+xL+20)/thermal_shape[0]*1.5
            xL, xR = xL-v_dis*0.7/e**ori_factor, xR-v_dis*0.7/e**ori_factor
            yL, yR = yL+size_factor*100, yR+size_factor*100
            
        transformed_keypoints.append(((int(xL),int(yL)), (int(xR),int(yR))))
        
        # Find lip keypoints
        lip_dist = abs(keypoints_eyes[0][i][1]-keypoints_eyes[1][i][1])*1.5
        lips_keypoints.append([(int(keypoints_eyes[0][i][1]),int(keypoints_eyes[0][i][0]+0.5*lip_dist)), (int(keypoints_eyes[1][i][1]), int(keypoints_eyes[1][i][0]+lip_dist))])     
        
    return forehead_keypoints, transformed_keypoints, lips_keypoints

def detect_roi_coords(engine, img):
    
    # Posenet getting keypoints
    pil_image = img
    # engine = PoseEngine('./forehead_detection/models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
    poses, keypoints, img, inference_time = engine.DetectPosesInImage(np.uint8(pil_image),0)
    # Getting forehead coordinates for 10 people in frame (NOTE: posenet currently processes a maximum of 10 people in the frame)
    keypoints_eyes = (keypoints[:,1,:], keypoints[:,2,:])
    keypoints_shoulders = (keypoints[:,5,:], keypoints[:,6,:])
    forehead_keypoints = []
    face_keypoints = []
    roi_keypoints = []
    shldr_keypoints = []
    for i in range(10):
        dist = abs(keypoints_eyes[0][i][1]-keypoints_eyes[1][i][1])
        # Format: ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y+offset)) offset is the distance wanted above the eyes.
        forehead_keypoints.append([(int(keypoints_eyes[0][i][1]+dist*0.5),int(keypoints_eyes[0][i][0]+dist*0.2)), (int(keypoints_eyes[1][i][1]-dist*0.5), int(keypoints_eyes[1][i][0]-dist*1.5))])     
        # Face coords for HR detection
        eye_dist = abs(keypoints_eyes[0][i][1]-keypoints_eyes[1][i][1])
        roi_keypoints.append([(int(keypoints_eyes[0][i][1])+ 0.2*eye_dist,int(keypoints_eyes[0][i][0]+0.2*eye_dist)),
                               (int(keypoints_eyes[1][i][1])-0.2*eye_dist, int(keypoints_eyes[1][i][0]+1*eye_dist))]) # for smaller region of face used for HR ets
        face_keypoints.append([(int(keypoints_eyes[0][i][1])-0.5*eye_dist,int(keypoints_eyes[0][i][0]-eye_dist)),
                               (int(keypoints_eyes[1][i][1])+0.5*eye_dist, int(keypoints_eyes[1][i][0]+2*eye_dist))])
        
        # Chest coords
        shoulder_dist = abs(keypoints_shoulders[0][i][1]-keypoints_shoulders[1][i][1])
        shldr_keypoints.append([(int(keypoints_shoulders[0][i][1]),int(keypoints_shoulders[0][i][0])),
                               (int(keypoints_shoulders[1][i][1]), int(keypoints_shoulders[1][i][0]+1.5*shoulder_dist))])
        
    return face_keypoints, roi_keypoints, shldr_keypoints

def draw_bounding_box(input_img, forehead_keypoints, temps):
    
    img_in = input_img
    img_out = deepcopy(img_in)
    
    COLOR_RED = (255,0, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_GR = (0,255,0)
    
    thresh = 37.4
    for i in range(10):
        temp = temps[i]
        ftemp = temp*9/5+32
        if forehead_keypoints[i][0][1]==0:
            continue
        if temp>=thresh:
            cv2.rectangle(img_out, forehead_keypoints[i][0], forehead_keypoints[i][1], COLOR_RED, 2)
            text = '{:.1f}'.format(temp)
            ftemp = '{:.1f}'.format(ftemp)
            txt = 'FEVER: ' + text + 'C/' + ftemp + 'F'
            cv2.putText(img_out, txt, (forehead_keypoints[i][1][0], forehead_keypoints[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        COLOR_RED, 2)
        else:
            cv2.rectangle(img_out, forehead_keypoints[i][0], forehead_keypoints[i][1], COLOR_GR, 2)
            text = '{:.1f}'.format(temp)
            ftemp = '{:.1f}'.format(ftemp)
            txt = str(text) + 'C/' + ftemp + 'F'
            if temp != 0:
                cv2.putText(img_out, txt, (forehead_keypoints[i][1][0], forehead_keypoints[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        COLOR_GR, 2)
    return img_out

def draw_lip_bounding_box(input_img, lip_keypoints, cyan_pred):
    
    img_in = input_img
    img_out = deepcopy(img_in)
    
    COLOR_RED = (255,0, 0)
    COLOR_GR = (0,255,0)
    
    for i in range(10):
        if lip_keypoints[i][0][1]==0 or len(cyan_pred)<=i:
            continue
        if cyan_pred[i][0] == 1:
            cv2.rectangle(img_out, lip_keypoints[i][0], lip_keypoints[i][1], COLOR_RED, 2)
            txt = 'CYANOSIS'
            cv2.putText(img_out, txt, (lip_keypoints[i][1][0], lip_keypoints[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        COLOR_RED, 2)
        else:
            cv2.rectangle(img_out, lip_keypoints[i][0], lip_keypoints[i][1], COLOR_GR, 2)
            txt = 'NORMAL'
            cv2.putText(img_out, txt, (lip_keypoints[i][1][0], lip_keypoints[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        COLOR_GR, 2)
    return img_out
        


