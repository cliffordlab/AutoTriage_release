#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Author: Jacob Zelko (alias TheCedarPrince)
Date: June 6th, 2018
License:
BSD-3 License
Copyright [2018] [Clifford Lab]
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from datetime import datetime
from threading import Thread
from time import sleep

from Adafruit_DHT import read_retry, DHT22

class SensorToolbox:

    """
    A superclass that provides subclasses to interface with hardware sensors.
    SensorTool enables one to interface with temperature and humidity
    sensors currently using the Adafruit_DHT module. More sensors and modules
    can be added by creating additional subclasses for each sensor in this
    superclass.
    """
    class DHT22Tool:

        """
        Samples temperature in current environment using the Adafruit DHT22.
        Class provides an interface to the DHT22 with associated methods to poll
        the current environment's temperature and humidity. By default,
        temperature is returned in celsius and humidity is given as a percent.
        Attributes
        ----------
        pin: int
            Expects an integer designating data output pin.
        sensor : DHT22 Object
            Expects a DHT22 Object designating the sensor used.
        Example
        -------
        >>> dht22_sensor = DHT22Tool(pin = 4)
        >>> dht22_sensor.detect(poll_time = 2)
        >>> print(dht22_sensor.return_temperature())
            23.8
        >>> print(dht22_sensor.return_humidity())
            64.0
        """

        def __init__(self, pin):
            """Instantiates sensor to interface with and pin for data output"""
            self.sensor = DHT22
            self.pin = pin

        def _detect(self, celsius, fahrenheit, poll_time):
            """
            Thread for detecting temperature.
            Uses Adafruit_DHT module to detect temperature of environment.
            Properties
            ----------
            Inherits **kwargs from detect method.
            """
            while True:
                self.humidity, self.temperature = read_retry(self.sensor,
                                                             self.pin)

                if celsius:
                    pass
                else:
                    self.temperature = self.temperature * 9/5.0 + 32
                sleep(poll_time)

        def detect(self, celsius=True, fahrenheit=False, poll_time=5):
            """
            Initiates _detect daemon thread
            Parameters
            ----------
            celsius : boolean
                Accepts a boolean expression to return temperatue in celsius
            fahrenheit : boolean
                Accepts a boolean expression to return temperature in fahrenheit
            poll_time : int
                Accepts an integer denoting how often to poll environment in
                seconds
            """
            detect_thread = Thread(target=self._detect,
                                 kwargs={"celsius": celsius,
                                         "fahrenheit": fahrenheit,
                                         "poll_time": poll_time})
            detect_thread.daemon = True
            detect_thread.start()

        def return_temperature(self):
            """Returns a string denoting temperature"""
            return str(self.temperature)

        def return_humidity(self):
            """Returns a string denoting current humidity"""
            return str(self.humidity)