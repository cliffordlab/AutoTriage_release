# AutoTriage
AutoTriage - An Open Source Edge Computing Raspberry Pi-based Clinical Screening System

## Hardware Requirements
1. RaspPi 4 4G
2. FLIR Lepton 3.5 thermal camera 
3. PureThermal i/o borad for Lepton camera
4. PiCamera V2
5. Google Coral usb TPU

**FLIR Lepton camera setup:** Make sure it has a firmware version later than 1.2.2, check https://github.com/groupgets/purethermal1-firmware for more details. 

## Software Prerequisites
python3, gstreamer, v4l2, tflite, opencv, Adafruit_DHT are required. 
Install latest NOOBs, enable camera.  
Use the script: `setup.sh` to install the prerequisites.

## Execution 
run `measure.py` for detecting cyanosis and temperature and displaying. After forced stop/ errors, `kill -9 $(pidof gst-launch-1.0)` should be used to reset the thermal camera before the next run. 

Environmental temperature and humidity can measured with `dht22_sensor_toolbox.py`.

Heart rate and respiratory effort estimation can be found in `./HeartRate_Respiration`, where `*_realtime_vis` provides read-time visualization of the detected areas, the `*_high_fs` scripts only plot the first image captured with detection, but provides higher sampling frequency. 


## Temperature Calibration 
As described in the manuscript, the FLIR Lepton thermal camera needs to be calibrated in the actual operating 
environment.  

Shell script `capture` can be used like `capture <name of the image>` to capture a single thermal picture with pixels 
being raw value (output temperature measured by 
the Lepton in Kelvin * 100), saved in `.pnm` format. After taking multiple pictures of the stable heat source at 
multiple known temperature, we can fit the uncalibrated output of the Lepton to the ground truth temperatures with a 
robust regression. This fitting process can be done with any tool you like, but here is an example code with Python:
```python
from sklearn.linear_model import HuberRegressor
import numpy as np
import matplotlib.pyplot as plt

# measurements are the average uncalibrated temperature (output) of the ROI (heat source) in the frame (i.e. average 
# pixel value of the roi)
# true_temp are the ground truth temperatures of the heat source
huber = HuberRegressor().fit(np.vstack([measurements, 30000*np.ones(len(measurements))]).transpose(), true_temp)
print(huber.coef_)
xp = np.linspace(30700, 31300, 1000)
yp = huber.predict(np.vstack([xp, 30000*np.ones(len(xp))]).transpose())
plt.scatter(measurements,true_temp, color='b')
```

After getting the coefficients of the fitted line, you can replace the coefficients on line `temp = 0.0113*temp - 313
.383` in `measure.py`with the new ones. (remember to multiply the slope with 30000 if the above code is used)

## Citation
Please cite the following when using:
```
article {Hegde2020.04.09.20059840,
	author = {Hegde, Chaitra and Jiang, Zifan and Suresha, Pradyumna Byappanahalli and Zelko, Jacob and Seyedi, Salman and Smith, Monique A and Wright, David W and Kamaleswaran, Rishikesan and Reyna, Matt A. and Clifford, Gari D},
	title = {AutoTriage - An Open Source Edge Computing Raspberry Pi-based Clinical Screening System},
	elocation-id = {2020.04.09.20059840},
	year = {2020},
	doi = {10.1101/2020.04.09.20059840},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2020/04/30/2020.04.09.20059840},
	eprint = {https://www.medrxiv.org/content/early/2020/04/30/2020.04.09.20059840.full.pdf},
	journal = {medRxiv}
}
```
