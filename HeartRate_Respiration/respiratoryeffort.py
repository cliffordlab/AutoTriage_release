# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:00:59 2020

@author: chait, Zifan
"""

import cv2
import numpy as np
from scipy.signal import detrend, firwin, welch, hilbert
from copy import deepcopy
import matplotlib.pyplot as plt
import time
from spectrum import *
import argparse
from scipy import interpolate

# set measurement time in second
parser = argparse.ArgumentParser(description='Set measurement time')
parser.add_argument('--t', type=int, default=10, help='set the length of time for one measurement')
parser.add_argument('--fs', type=int, default=5, help='sampling freq for frame difference')
args = parser.parse_args()

measurement_time = args.t

############################## Viola-Jones Face Detection - Change to PoseNet #############################
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def viola_jones(img):
    img1 = deepcopy(img)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #img = cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
        #img = cv2.rectangle(img1,(int(x+w*0.25),int(y+h-0.05*h)),(int(x+w*0.75),int(y+h*1.3)), (0,255,0),2)
        img = cv2.rectangle(img1,(int(x-w*0.5),int(y+1.5*h)),(int(x+w*2*0.7),int(y+h*3)), (0,255,0),2)

    return img1, int(x+w*0.25), int(y+h-0.05*h), int(x+w*0.75), int(y+h*1.3), x, y, w, h


################################### First difference between each pixel ##############################
def first_dif(roi, prev_roi):
    diff = roi - prev_roi
    avg_diff_R = np.mean(diff[:,:,2])
    avg_diff_G = np.mean(diff[:,:,1])
    avg_diff_B = np.mean(diff[:,:,0])
    
    avg = (avg_diff_R+avg_diff_G+avg_diff_B)/3
    
    return avg

def pixel_diff(roi, prev_roi):
    # Make images same size - pad smaller img
    x1, y1, _ = roi.shape
    x2, y2, _ = prev_roi.shape

    new_roi = np.zeros((max(x1,x2), max(y1,y2),3))
    new_prev_roi = np.zeros((max(x1,x2), max(y1,y2),3))
    
    roi_x_pad = new_roi.shape[0] - roi.shape[0]
    roi_y_pad = new_roi.shape[1] - roi.shape[1]
    prev_roi_x_pad = new_prev_roi.shape[0] - prev_roi.shape[0]
    prev_roi_y_pad = new_prev_roi.shape[1] - prev_roi.shape[1]
    
    if roi_x_pad%2 == 0 and roi_y_pad%2 == 0:
        #new_roi[int(roi_x_pad/2):new_roi.shape[0]-int(roi_x_pad/2),int(roi_y_pad/2):-int(roi_y_pad/2),:] = roi
        new_roi[int(roi_x_pad/2):int(roi_x_pad/2)+x1, int(roi_y_pad/2):int(roi_y_pad/2)+y1, :] = roi
        
    elif roi_x_pad%2 == 0 and roi_y_pad%2 != 0:
        #new_roi[int(roi_x_pad/2):-int(roi_x_pad/2),int(np.floor(roi_y_pad/2)):-int(np.floor(roi_y_pad/2)+1),:] = roi
        new_roi[int(roi_x_pad/2):int(roi_x_pad/2)+x1, int(np.floor(roi_y_pad/2)):int(np.floor(roi_y_pad/2))+y1, :] = roi
    
    elif roi_x_pad%2 != 0 and roi_y_pad%2 == 0:
        #new_roi[int(np.floor(roi_x_pad/2)):-int(np.floor(roi_x_pad/2)+1),int(roi_y_pad/2):-int(roi_y_pad/2),:] = roi
        new_roi[int(np.floor(roi_x_pad/2)):int(np.floor(roi_x_pad/2))+x1, int(roi_y_pad/2):int(roi_y_pad/2)+y1, :] = roi
    
    elif roi_x_pad%2 != 0 and roi_y_pad%2 != 0:
        #new_roi[int(np.floor(roi_x_pad/2)):-int(np.floor(roi_x_pad/2)+1),int(np.floor(roi_y_pad/2)):-int(np.floor(roi_y_pad/2)+1),:] = roi
        new_roi[int(np.floor(roi_x_pad/2)):int(np.floor(roi_x_pad/2))+x1, int(np.floor(roi_y_pad/2)):int(np.floor(roi_y_pad/2))+y1, :] = roi



    if prev_roi_x_pad%2 == 0 and prev_roi_y_pad%2 == 0:
        #new_prev_roi[int(prev_roi_x_pad/2):-int(prev_roi_x_pad/2),int(prev_roi_y_pad/2):-int(prev_roi_y_pad/2),:] = prev_roi
        new_prev_roi[int(prev_roi_x_pad/2):int(prev_roi_x_pad/2)+x2, int(prev_roi_y_pad/2):int(prev_roi_y_pad/2)+y2, :] = prev_roi

    elif prev_roi_x_pad%2 == 0 and prev_roi_y_pad%2 != 0:
        #new_prev_roi[int(prev_roi_x_pad/2):-int(prev_roi_x_pad/2),int(np.floor(prev_roi_y_pad/2)):-int(np.floor(prev_roi_y_pad/2)+1),:] = prev_roi
        new_prev_roi[int(prev_roi_x_pad/2):int(prev_roi_x_pad/2)+x2, int(np.floor(prev_roi_y_pad/2)):int(np.floor(prev_roi_y_pad/2))+y2, :] = prev_roi

    elif prev_roi_x_pad%2 != 0 and prev_roi_y_pad%2 == 0:
        #new_prev_roi[int(np.floor(prev_roi_x_pad/2)):-int(np.floor(prev_roi_x_pad/2)+1),int(prev_roi_y_pad/2):-int(prev_roi_y_pad/2),:] = prev_roi
        new_prev_roi[int(np.floor(prev_roi_x_pad/2)):int(np.floor(prev_roi_x_pad/2))+x2, int(prev_roi_y_pad/2):int(prev_roi_y_pad/2)+y2, :] = prev_roi
    
    elif prev_roi_x_pad%2 != 0 and prev_roi_y_pad%2 != 0:
        #new_prev_roi[int(np.floor(prev_roi_x_pad/2)):-int(np.floor(prev_roi_x_pad/2)+1),int(np.floor(prev_roi_y_pad/2)):-int(np.floor(prev_roi_y_pad/2)+1),:] = prev_roi
        new_prev_roi[int(np.floor(prev_roi_x_pad/2)):int(np.floor(prev_roi_x_pad/2))+x2, int(np.floor(prev_roi_y_pad/2)):int(np.floor(prev_roi_y_pad/2))+y2, :] = prev_roi
            
    return abs(new_roi - new_prev_roi)

def diff_avg(diff_frames, fs):
    #avg = 0
    means = []
    for i in range(1,len(diff_frames)): # To remove diff of first frame with itself
        means.append(np.mean(diff_frames[i]))
        #avg += np.mean(diff_frames[i])
    #avg = avg/(len(diff_frames)-1)
    hamming_coeffs_resp = firwin(95, [0.05/fs, 2.0/fs], pass_zero=False) # Bandpass filter
                                                                      # numtaps is window length
    hamming = np.convolve(hamming_coeffs_resp, means, mode='full')
    return hamming

################################# Main ################################
cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("Could not open video device")

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Get framerate
fps = round(cap.get(cv2.CAP_PROP_FPS))

frame_count = 0
diff_frames = []
sampling_freq = args.fs  
sampling_num = fps/sampling_freq  # number of frames between the two frames used for diff calculation
samp_count = 0
pic_count=0
time.sleep(0.1)
start = time.time()
while(True):
    ret, frame = cap.read()
    frame_bb, roi_x, roi_y, roi_w, roi_h, fx, fy, fw, fh = viola_jones(frame)
    frame_roi = frame[roi_y:roi_h, roi_x:roi_w]
    if samp_count == sampling_num:
        if frame_count < measurement_time*sampling_freq: 
            if frame_count == 0:
                pix_diff = np.zeros_like(frame_roi)
            else:
                pix_diff = pixel_diff(frame_roi, prev_frame_roi)
            diff_frames.append(pix_diff)
            frame_count += 1
            prev_frame_roi = frame_roi
        if frame_count == measurement_time*sampling_freq:
            # estimate real fps
            real_sampling_freq = frame_count / (time.time()-start)
            print(real_sampling_freq)
            resp_effort = diff_avg(diff_frames, real_sampling_freq)
            f, PSD = welch(resp_effort, fs=real_sampling_freq, window='hamming', nperseg=len(resp_effort))
            plt.figure(); plt.plot(f,PSD); plt.xlabel('Freq in Hz'); plt.show()
            ii = np.argmax(PSD)
            print(max(PSD),f[ii])
            frame_count = 0
            diff_frames = []
            start = time.time()
        samp_count = 0
    else:
        samp_count += 1
        pic_count+=1
    
    cv2.imshow('HR', frame_bb)
    if pic_count == 100:
        print('done')
        temp = frame_bb
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()