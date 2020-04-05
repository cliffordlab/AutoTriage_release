from picamera import PiCamera
import time
import subprocess
from PIL import Image
import numpy as np
import sys, os, io
sys.path.append('./forehead_detection')
from detect_forehead import *
from cyanosis_detection import predict_cyanosis 
from pose_engine import PoseEngine
import matplotlib.pyplot as plt
from joblib import load 

use_cv2 = False
if use_cv2:
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, False)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y16 '))
else:
    cmd = "gst-launch-1.0 v4l2src num-buffers=-1 device=/dev/video1 ! video/x-raw,format=GRAY16_LE ! pnmenc ! multifilesink location=./tmp/test.pnm".split()
    FLIR = subprocess.Popen(cmd)
    time.sleep(2)
camera = PiCamera(resolution=(1640,1232), framerate=40)
# test camera mode for cyanosis
camera.awb_mode = 'sunlight'
engine = PoseEngine('./forehead_detection/models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
stream = io.BytesIO()
# setting up plot canvas
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig.canvas.draw()
plt.show(block=False)

svc_clf = load('svc_model.joblib') 

# about 1hr
for i in range(10000):
    start = time.time()
    # normal image
    camera.capture(stream, format='jpeg', use_video_port=True)
    img = Image.open(stream)
    img = img.resize((641,481), Image.NEAREST)
    # get cordinates of faces
    # camera fov factor:
    # horizontal: np.tan(57/2*np.pi/180)/np.tan(31.1*np.pi/180) = 0.9
    # diagonal of lepton: 71degree, vertical=np.sqrt(71.3293**2 -54.295**2)=46.259
    # vertical: 0.46259/np.tan(24.4*np.pi/180) = 1.012
    coords, transform_coords, lip_coords = forehead_coords(engine, img, (160,120), 0.9, 1.012) 
    # predict cyanosis
    cyanosis_preds = predict_cyanosis(svc_clf, np.array(img), lip_coords)
    
    if use_cv2:
        for n in range(2):
            cap.grab()
        _, flir_im = cap.read()
        flir_im = flir_im[:120,:]
    else:
        flir_im = np.array(Image.open('./tmp/test.pnm'))[:120,:]
    temps = []
    for k in transform_coords:
        xL = np.clip(k[0][0],0,160)
        yL = np.clip(k[0][1],0,120)
        xR = np.clip(k[1][0],0,160)
        yR = np.clip(k[1][1],0,120)
        roi = flir_im[xR:xL,yR:yL]
        temp = np.sort(roi.flatten())[-5:].mean()
        R = 580061
        O = 25113
        temp = 1428/np.log(R/(temp-O)+1) -273.15 
        temps.append(temp)
    # testing for thermal imaging
    flir_im = 1428/np.log(R/(flir_im-O)+1) -273.15
    flir_out = draw_bounding_box(flir_im, transform_coords, temps)
    # plotting
    img_out = draw_bounding_box(np.array(img.convert('RGB')), coords, temps)
    img_out = draw_lip_bounding_box(img_out, lip_coords, cyanosis_preds) 
    if i==0:
        plot = ax.imshow(img_out)
        bkg = fig.canvas.copy_from_bbox(ax.bbox)
    else:
        plot.set_data(img_out)
        fig.canvas.restore_region(bkg)
        ax.draw_artist(plot)
        fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()
    stream.seek(0)
    stream.truncate()
stream.close()
if use_cv2:
    cap.release()
else:
    FLIR.kill()
