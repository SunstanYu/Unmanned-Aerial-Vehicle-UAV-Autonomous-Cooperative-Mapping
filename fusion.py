import numpy as np
import cv2
from matplotlib import pyplot as plt

import imregpoc


def calWeight(d, k):
    '''
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    '''

    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y


def imgFusion(img1, img2, coordinates_start, coordinates_end):
    '''
    图像加权融合
    :param img1:
    :param img2:
    :param overlap: 重合长度
    :param left_right: 是否是左右融合
    :return:
    '''
    # 这里先暂时考虑平行向融合
    # img1 = (img1 - img1.min()) / img1.ptp()
    # img2 = (img2 - img2.min()) / img2.ptp()
    x_o, y_o = coordinates_start
    x_t, y_t = coordinates_end
    x_t += 1
    y_t += 1
    overlap_x = x_t - x_o
    overlap_y = y_t - y_o
    # 水平
    w_h = calWeight(overlap_x, 0.05)  # k=5 这里是超参
    # 垂直
    w_v = calWeight(overlap_y, 0.05)
    img_new = img1.copy()
    col, row = img1.shape
    w_expand = np.tile(w_h, (overlap_y, 1))  # 权重扩增
    # black_pixel = img1 == 0
    img_new[y_o:y_t, x_o:x_t] = (1 - w_expand) * img1[y_o:y_t, x_o:x_t] + w_expand * img2
    # img_new = np.uint8(img_new * 255)
    return img_new


if __name__ == "__main__":
    img1 = cv2.imread("data/map1_1.jpg", 0)
    img2 = cv2.imread("data/map1_2.jpg", 0)
    height, width = img1.shape
    temp = img1[height - 5:, width - 5:]
    # plt.figure()
    # plt.imshow(temp, vmin=temp.min(), vmax=temp.max(), cmap='gray')
    # plt.show()
    img_new = imgFusion(img2, temp, (26,26),(31,31))
    plt.figure()
    plt.imshow(img_new, vmin=img_new.min(), vmax=img_new.max(), cmap='gray')
    plt.show()
    cv2.imwrite('data/test_new3.jpg', img_new)
