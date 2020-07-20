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

## Citation
Please cite the following when using:
'''
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
'''
