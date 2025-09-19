#!/usr/bin/env python3

import image
import time
import cv2
from link import Serial, Udp, Tcp
from infrared import Infrared

infrared = Infrared()
serial = Serial()
udp = Udp("192.168.2.72")
tcp = Tcp("192.168.2.72")
image = image.Image()

port = serial

if __name__ == '__main__':
    while True:
        try:
            port.open()


            num = 115
            while (True):
                if num == 120:
                    num = 115
                tempMatrix = infrared.calculateCelsius(
                    infrared.parse(port.recv()))
                if (tempMatrix is None):
                    continue
                infrared.printTemp(tempMatrix)
                output = image.normalization(tempMatrix)
                image.show(output)
                cv2.imwrite(f"data/heatmap{num}.jpg", output)
                num += 1
        except Exception as e:
            print(e)
        finally:
            port.close()
            time.sleep(1)
