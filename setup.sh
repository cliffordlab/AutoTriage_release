#!/bin/bash
#install dependencies for AutoTriage

sudo apt-get update
root=$(pwd)

# libatlas
sudo apt-get install libatlas-base-dev
# posenet 
source ./forehead_detection/install_requirements.sh
# gstreamer
sudo apt-get install gstreamer1.0-tools
# opencv
sudo apt-get install python3-opencvS
# coral accelerator
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std
# tflite for py3.7, check https://www.tensorflow.org/lite/guide/python otherwise
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
# edgetpu api
sudo apt-get install python3-edgetpu
# scipy, sklearn, spectrum
pip3 install scipy
pip3 install sklearn
pip3 install spectrum

PATH=$PATH:$root
mkdir tmp 
mkdir data