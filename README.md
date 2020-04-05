# AutoTriage
Automatic triage for COVID-19 with minimal physical contact and attention

## Hardware requirement
1. RaspPi 4
2. FLIR Lepton camera 
3. PureThermal i/o borad for Lepton camera
4. PiCamera
5. Google Coral usb TPU

**FLIR Lepton camera setup:** Make sure it has a firmware version $\gep$ 1.2.2, check https://github.com/groupgets/purethermal1-firmware for more details. 

## Software prerequisite
python3, gstreamer, v4l2, tflite, opencv,...  
Install latest NOOBs, enable camera and ssh (optional).  
use the script: `setup.sh`.
