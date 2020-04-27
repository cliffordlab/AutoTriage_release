# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:31:22 2020

@author: chait, Zifan
"""

import cv2
import numpy as np
from scipy.signal import detrend, firwin, welch, hilbert
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
from scipy import interpolate
import argparse
from picamera import PiCamera
import io
from PIL import Image


# set measurement time in second
parser = argparse.ArgumentParser(description='Set measurement time')
parser.add_argument('--t', type=int, default=20, help='set the length of time for one measurement')
parser.add_argument('--total', type=int, default=60, help='set the total length of time for monitoring')
args = parser.parse_args()

measurement_time = args.t
measurement_time_agg = args.total

#################### Viola-Jones Face Detection - Change to PoseNet ##################
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def viola_jones(img):
    img1 = deepcopy(img)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces)==0:
        raise ValueError('No Face detected')
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img1,(int(x+w*0.2),int(y+h*0.5)),(int(x+w*0.8),int(y+h*0.8)), (0,255,0),2)
        
    return img1, int(x+w*0.2), int(y+h*0.5), int(x+w*0.8), int(y+h*0.8), x, y, w, h

################################# Illumination Rectification ###############################
def illuminationRectification(h, mu, g_face, g_bg):
    g_ir = g_face - h*g_bg
    h = h + mu*np.dot(g_ir,g_bg)/np.dot(g_bg.conj().T,g_bg)
    
    return g_ir, h

################################# Non-rigid motion estimation ################################
def nrme(g_ir, fps):
    # divide into m segments. Let m = framerate. Therefore each segment is 1 sec long
    fps =round(fps)
    count = 0
    seg = []
    g_ir_seg = []
    for i in range(len(g_ir)):
        if count != fps:
            seg.append(g_ir[i])
            count += 1
        else:
            g_ir_seg.append(np.array(seg))
            seg = []
            count = 0

    # Get standard deviation of each segment
    std = []
    for i in range(len(g_ir_seg)):
        std.append(np.std(g_ir_seg[i]))
    print(len(g_ir_seg))
    # Remove 20% of the segments with highest standard deviation
    percent_to_num = int(np.ceil(0.2*len(g_ir_seg)))
    top_5_std = np.argsort(std)[-percent_to_num:]
    new_g_ir = []
    for i in range(len(g_ir_seg)):
        if i in top_5_std:
            pass
        else:
            new_g_ir.append(g_ir_seg[i])
            
    new_g_ir = np.array(new_g_ir)
    #print(np.shape(new_g_ir))
    new_g_ir = new_g_ir.reshape(new_g_ir.shape[0]*new_g_ir.shape[1])
    
    return new_g_ir

################################# Moving average filter ###############################
def moving_avg(signal, N): ########## CHECK, N is window length
    cumsum, moving_aves = [0], []
    for i, x in enumerate(signal, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return moving_aves

# For illumination rectification
def illum_rect(frame_roi, frame, fx, fy, fw, fh):
    g_face = np.mean(frame_roi[:,:,1].reshape((frame_roi[:,:,1].shape[0]*frame_roi[:,:,1].shape[1])))
    pixels_bg = frame[0:fy,0:fx,1].reshape(frame[0:fy,0:fx,1].shape[0]*frame[0:fy,0:fx,1].shape[1])
    pixels_bg = np.concatenate((pixels_bg,frame[fx:fx+fw,0:fy,1].reshape(frame[fx:fx+fw,0:fy,1].shape[0]*frame[fx:fx+fw,0:fy,1].shape[1])))
    pixels_bg = np.concatenate((pixels_bg,frame[0:fy,fx+fw:-1,1].reshape(frame[0:fy,fx+fw:-1,1].shape[0]*frame[0:fy,fx+fw:-1,1].shape[1])))
    pixels_bg = np.concatenate((pixels_bg,frame[fy:fy+fh,0:fx,1].reshape(frame[fy:fy+fh,0:fx,1].shape[0]*frame[fy:fy+fh,0:fx,1].shape[1])))
    pixels_bg = np.concatenate((pixels_bg,frame[fy:fy+fh,fx+fw:-1,1].reshape(frame[fy:fy+fh,fx+fw:-1,1].shape[0]*frame[fy:fy+fh,fx+fw:-1,1].shape[1])))
    pixels_bg = np.concatenate((pixels_bg,frame[fy+fh:-1,0:fx,1].reshape(frame[fy+fh:-1,0:fx,1].shape[0]*frame[fy+fh:-1,0:fx,1].shape[1])))
    pixels_bg = np.concatenate((pixels_bg,frame[fy+fh:-1,fx:fx+fw,1].reshape(frame[fy+fh:-1,fx:fx+fw,1].shape[0]*frame[fy+fh:-1,fx:fx+fw,1].shape[1])))
    pixels_bg = np.concatenate((pixels_bg,frame[fy+fh:-1,fx+fw:-1,1].reshape(frame[fy+fh:-1,fx+fw:-1,1].shape[0]*frame[fy+fh:-1,fx+fw:-1,1].shape[1])))
    g_bg = np.mean(pixels_bg)
    return g_face, g_bg

################################# Open camera (webcam, RPi cam) ######################################
# cap = cv2.VideoCapture(0)
# if not (cap.isOpened()):
#     print("Could not open video device")
# # Set resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# # Get framerate
# fps = round(cap.get(cv2.CAP_PROP_FPS))

# Picamera
camera = PiCamera(resolution=(640, 480), framerate=40)
fps = 6
stream = io.BytesIO()
# setting up plot canvas
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# fig.canvas.draw()
# plt.show(block=False)

frame_count = 0
G_bg = []
G_face = []
t_list = []
h = 0.5
mu = 0.1
view_g_bg = []
view_g_face = []
view_g_ir = []
H = []
hr_count = 0
mean_HR = []
RR = []
time.sleep(0.1)
start = time.time()
flag = 0
while(True):
    for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
            frame = np.array(Image.open(stream))
            frame_bb, roi_x, roi_y, roi_w, roi_h, fx, fy, fw, fh = viola_jones(frame)
            frame_roi = frame[roi_y:roi_h, roi_x:roi_w]
            g_face, g_bg = illum_rect(frame_roi, frame, fx, fy, fw, fh)
            G_bg.append(g_bg)
            G_face.append(g_face)
            frame_count += 1
            hr_count += 1
            t_list.append(time.time()-start)
            stream.seek(0)
            stream.truncate()
            if flag==0:
                plt.imshow(frame_bb)
                plt.pause(0.001)
#                bkg = fig.canvas.copy_from_bbox(ax.bbox)
#             else:
#                 plot.set_data(frame_bb)
#                 fig.canvas.restore_region(bkg)
#                 ax.draw_artist(plot)
#                 fig.canvas.blit(ax.bbox)
#             fig.canvas.flush_events()
            stream.seek(0)
            stream.truncate()
            flag = 1
            if frame_count >= fps*measurement_time:
                break

    print('time elapsed since last measurement: ' + str(time.time()-start))
    print('Avg sampling freq: ' + str(frame_count/(time.time()-start))) 
    # Illumination rectification
    g_ir, h = illuminationRectification(h, mu, np.array(G_face), np.array(G_bg))
    # resample g_ir through interpolation
    fcubic = interpolate.interp1d(np.array(t_list), g_ir)
    t_new = np.linspace(t_list[0], t_list[-1], len(g_ir))
    g_ir_new = fcubic(t_new)
    fps = frame_count / t_list[-1]

    #view_g_ir.append(g_ir)
    # Non rigid motion estimation
    nrme_inpt = np.array(g_ir)
    #nrme_inpt = nrme_inpt.reshape(nrme_inpt.shape[0]*nrme_inpt.shape[1])
    nrme_g_ir = nrme(nrme_inpt, round(fps))
    # Temporal Filtering
    detrend_g_ir = detrend(nrme_g_ir, type='constant') # Detrending filter
    movingavg_g_ir = moving_avg(detrend_g_ir, N=7) # Moving average filter
    f_nyquist = fps/2
    hamming_coeffs = firwin(95, [0.7/f_nyquist, 2/f_nyquist], pass_zero=False) # Bandpass filter
                                                                  # numtaps is window length
    #hamming_g_ir = np.convolve(hamming_coeffs/hamming_coeffs.sum(), movingavg_g_ir, mode='valid')
    hamming_g_ir = np.convolve(hamming_coeffs, movingavg_g_ir, mode='full')
    RR.append(hamming_coeffs)
    # Convert signal to frequency domain
    #fft_g_ir = abs(np.fft.fft(hamming_g_ir))
    # Get PSD using Welch's method
    f, PSD_g_ir = welch(hamming_g_ir, fs=fps, nperseg=len(hamming_g_ir))
    f_m = moving_avg(f, N=5)
    PSD_m = moving_avg(PSD_g_ir, N=7)
    # Find heartrate
    ind = np.argmax(PSD_g_ir)
    f_hr = f[ind] #- 1/fps
    HR = 60*f_hr
    mean_HR.append(HR)
    print(HR)
        
    # reinit
    frame_count = 0
    hr_count += 1
    G_bg = []
    G_face = []
    view_g_ir = []
    start = time.time()
    t_list = []
    flag=0
        
        
    if hr_count == fps*measurement_time_agg:
        final_HR = np.mean(mean_HR)
        print('mean HR: ', final_HR)
        hr_count = 0
        mean_HR = []
        h = 0.5
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.close()
cap.release()
cv2.destroyAllWindows()
