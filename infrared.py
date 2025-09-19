#!/usr/bin/env python3

import struct
import numpy as np


class Infrared:
    def __init__(self):
        self.cache = bytearray()

    def calculateCelsius(self, frame):
        if (frame is None):
            return None
        tempMatrix = np.zeros([32, 32], dtype=float)
        for i in range(int(len(frame) / 2)):
            temp = struct.unpack("H", frame[i*2:i*2+2])[0] / 10 - 273.15
            y = int(i / 32)
            x = int(i % 32)
            tempMatrix[y][x] = temp
        return tempMatrix

    def printTemp(self, tempMatrix):
        if (tempMatrix is None):
            return None
        print("\n\n")
        print('\n'.join([' '.join(['{:.1f}'.format(item)
                                   for item in row]) for row in tempMatrix]))

    def parse(self, buffer):
        if (buffer is None):
            return None
        self.cache.extend(buffer)
        startToken = b"---------------[frame start]---------------"
        endToken = b"----------------[frame end]----------------"
        indexTokenStart = self.cache.find(startToken)
        indexTokenEnd = self.cache.find(endToken, indexTokenStart)
        startOfFrame = indexTokenStart + len(startToken)
        frameLen = indexTokenEnd - startOfFrame
        if (indexTokenStart >= 0 and indexTokenEnd > indexTokenStart and frameLen == 32*32*2):
            frame = bytes(self.cache[startOfFrame:startOfFrame+frameLen])
            self.cache = self.cache[indexTokenEnd + len(endToken):]
            return frame
        elif (len(self.cache) > 32*32*2*2):
            self.cache = bytearray()
        return None
