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
import argparse
from scipy import interpolate
from picamera import PiCamera
import io, sys
from PIL import Image
sys.path.append('../forehead_detection')
from detect_forehead import *
from pose_engine import PoseEngine

# set measurement time in second
parser = argparse.ArgumentParser(description='Set measurement time')
parser.add_argument('--t', type=int, default=10, help='set the length of time for one measurement')
parser.add_argument('--fs', type=int, default=5, help='sampling freq for frame difference')
args = parser.parse_args()

measurement_time = args.t

# ############################## Viola-Jones Face Detection #############################
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
# mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

engine = PoseEngine('../forehead_detection/models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')

def viola_jones(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(int(x-w*0.5),int(y+1.5*h)),(int(x+w*2*0.7),int(y+h*3)), (0,255,0),2)

    return img, int(x+w*0.25), int(y+h-0.05*h), int(x+w*0.75), int(y+h*1.3), x, y, w, h


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
        new_roi[int(roi_x_pad/2):int(roi_x_pad/2)+x1, int(roi_y_pad/2):int(roi_y_pad/2)+y1, :] = roi
    
    elif roi_x_pad%2 != 0 and roi_y_pad%2 == 0:
        #new_roi[int(np.floor(roi_x_pad/2)):-int(np.floor(roi_x_pad/2)+1),int(roi_y_pad/2):-int(roi_y_pad/2),:] = roi
        new_roi[int(roi_x_pad/2):int(roi_x_pad/2)+x1, int(roi_y_pad/2):int(roi_y_pad/2)+y1, :] = roi
    
    elif roi_x_pad%2 != 0 and roi_y_pad%2 != 0:
        #new_roi[int(np.floor(roi_x_pad/2)):-int(np.floor(roi_x_pad/2)+1),int(np.floor(roi_y_pad/2)):-int(np.floor(roi_y_pad/2)+1),:] = roi
        new_roi[int(roi_x_pad/2):int(roi_x_pad/2)+x1, int(roi_y_pad/2):int(roi_y_pad/2)+y1, :] = roi



    if prev_roi_x_pad%2 == 0 and prev_roi_y_pad%2 == 0:
        #new_prev_roi[int(prev_roi_x_pad/2):-int(prev_roi_x_pad/2),int(prev_roi_y_pad/2):-int(prev_roi_y_pad/2),:] = prev_roi
        new_prev_roi[int(prev_roi_x_pad/2):int(prev_roi_x_pad/2)+x2, int(prev_roi_y_pad/2):int(prev_roi_y_pad/2)+y2, :] = prev_roi

    elif prev_roi_x_pad%2 == 0 and prev_roi_y_pad%2 != 0:
        #new_prev_roi[int(prev_roi_x_pad/2):-int(prev_roi_x_pad/2),int(np.floor(prev_roi_y_pad/2)):-int(np.floor(prev_roi_y_pad/2)+1),:] = prev_roi
        new_prev_roi[int(prev_roi_x_pad/2):int(prev_roi_x_pad/2)+x2, int(prev_roi_y_pad/2):int(prev_roi_y_pad/2)+y2, :] = prev_roi

    elif prev_roi_x_pad%2 != 0 and prev_roi_y_pad%2 == 0:
        #new_prev_roi[int(np.floor(prev_roi_x_pad/2)):-int(np.floor(prev_roi_x_pad/2)+1),int(prev_roi_y_pad/2):-int(prev_roi_y_pad/2),:] = prev_roi
        new_prev_roi[int(prev_roi_x_pad/2):int(prev_roi_x_pad/2)+x2, int(prev_roi_y_pad/2):int(prev_roi_y_pad/2)+y2, :] = prev_roi
    
    elif prev_roi_x_pad%2 != 0 and prev_roi_y_pad%2 != 0:
        #new_prev_roi[int(np.floor(prev_roi_x_pad/2)):-int(np.floor(prev_roi_x_pad/2)+1),int(np.floor(prev_roi_y_pad/2)):-int(np.floor(prev_roi_y_pad/2)+1),:] = prev_roi
        new_prev_roi[int(prev_roi_x_pad/2):int(prev_roi_x_pad/2)+x2, int(prev_roi_y_pad/2):int(prev_roi_y_pad/2)+y2, :] = prev_roi
            
    return abs(new_roi - new_prev_roi)

def diff_avg(diff_frames, fs):
    means = []
    for i in range(1,len(diff_frames)): # To remove diff of first frame with itself
        means.append(np.mean(diff_frames[i]))
    hamming_coeffs_resp = firwin(65, [0.05/(fs/2), 1.0/(fs/2)], pass_zero=False) # Bandpass filter
    hamming = np.convolve(hamming_coeffs_resp, means, mode='same')

    return hamming

################################# Main ################################
# Picamera
camera = PiCamera(resolution=(640, 480), framerate=40)
fps = 5
stream = io.BytesIO()
# setting up plot canvas
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig.canvas.draw()
plt.show(block=False)

frame_count = 0
diff_frames = []
sampling_freq = args.fs  
sampling_num = fps/sampling_freq  # number of frames between the two frames used for diff calculation
samp_count = 0
pic_count=0
time.sleep(0.1)
start = time.time()
flag=0
while(True):
    camera.capture(stream, format='jpeg', use_video_port=True)
    frame = np.array(Image.open(stream))
    # VJ
    # frame_bb, roi_x, roi_y, roi_w, roi_h, fx, fy, fw, fh = viola_jones(frame)
    # frame_roi = frame[roi_y:roi_h, roi_x:roi_w]
    # openpose
    _, _, chest_coords = detect_roi_coords(engine, frame)
    x1, y1 = chest_coords[0][0]
    x2, y2 = chest_coords[0][1]
    frame_roi = frame[x2:x1, y1:y2]
    
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
    
    chest_coords[0] = ((x1, y1), (x2, y2))
    frame_bb = draw_bounding_box(frame, chest_coords, np.zeros(10))
    if flag==0:
        plot = ax.imshow(frame_bb)
        bkg = fig.canvas.copy_from_bbox(ax.bbox)
    else:
        plot.set_data(frame_bb)
        fig.canvas.restore_region(bkg)
        ax.draw_artist(plot)
        fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()
    stream.seek(0)
    stream.truncate()
    
    if pic_count == 50:
        print('done')
        temp = frame_bb
    
    flag=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()