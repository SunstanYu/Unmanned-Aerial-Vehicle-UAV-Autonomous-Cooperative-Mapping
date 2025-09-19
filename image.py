#!/usr/bin/env python3

import cv2
import numpy as np


class Image:
    def normalization(self, matrix):
        vMax = matrix.max()
        vMin = matrix.min()
        [rows, cols] = matrix.shape
        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = (matrix[i][j] - vMin) / (vMax - vMin)
        imageGray = np.zeros(matrix.shape, dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                imageGray[i][j] = 255 - matrix[i][j] * 255
        return imageGray

    def show(self, imageGray, style=cv2.COLORMAP_RAINBOW, algorithm=cv2.INTER_LANCZOS4, scale=10):
        [rows, cols] = imageGray.shape
        imageResizeGray = cv2.resize(
            imageGray, (rows * scale, cols * scale), interpolation=algorithm)
        imageColor = cv2.applyColorMap(imageResizeGray, style)
        cv2.imshow("Infrared", imageColor)
        cv2.waitKey(10)
        if (cv2.getWindowProperty("Infrared", style) == 0):
            exit()
