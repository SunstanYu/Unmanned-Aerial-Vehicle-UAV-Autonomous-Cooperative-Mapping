import cv2
import numpy as np
import random
import os
import shutil

def clear_directory(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def add_gaussian_noise():
    mean = 0
    std_dev = 1
    noise = np.random.normal(mean, std_dev)
    noise = np.clip(noise, -0.5, 0.5)  # 将噪声限制在正负3以内的范围内
    return noise

def extract_and_rotate(image, rect_coords, rotation_range, output_txt):
    # 提取矩形区域
    x1, y1, x2, y2 = rect_coords
    rect_image = image[y1:y2, x1:x2]
    cv2.imwrite(f"data/mappiece/all.png", rect_image)
    height, width = rect_image.shape
    num = 0
    # 创建 txt 文件
    with open(output_txt, 'w') as f:
        # 遍历矩形区域
        for y in range(height-32):
           if random.randint(0, 15) <= 5:
              for x in range(width-32):
                # 以随机间隔选择图像
                if random.randint(0, 15) <= 5:
                    # 提取 32x32 大小的图像
                    patch = rect_image[y:y + 32, x:x + 32]
                    # 生成随机角度
                    angle = random.uniform(-rotation_range, rotation_range)
                    # 旋转图像
                    rotated_patch = rotate_image(patch, angle)
                    # 保存图像
                    output_path = f"data/mappiece/img{num}.png"
                    num+=1
                    cv2.imwrite(output_path, patch)
                    # 将图像坐标和变换角度写入 txt 文件
                    # f.write(f"{round(x+add_gaussian_noise(),2)},{round(y+add_gaussian_noise(),2)},{round(angle,2)}\n")
                    f.write(f"{round(x,2)},{round(y,2)},{round(angle,2)}\n")

def rotate_image(image, angle):
    # 图像中心点坐标
    center = (image.shape[1] // 2, image.shape[0] // 2)
    # 旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    if image.shape[0] != 32 or image.shape[1] != 32:
        print(1)
    # 执行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

# 示例用法
image = cv2.imread('data/map1_down.jpg', cv2.IMREAD_GRAYSCALE)
rect_coords = (20, 20, 80, 80)  # 矩形区域左上角和右下角坐标
rotation_range = 20  # 旋转角度范围
output_txt = 'data/mappiece/labels.txt'  # 输出文本文件名
# 示例用法
folder_path = "data/mappiece"
clear_directory(folder_path)

extract_and_rotate(image, rect_coords, rotation_range, output_txt)
# print(add_gaussian_noise())