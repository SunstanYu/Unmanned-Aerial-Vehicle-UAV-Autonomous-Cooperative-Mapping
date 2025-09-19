import random

import cv2
import numpy as np
import image_process
import matplotlib.pyplot as plt
import judge
import SSIM
import PSNR
import imregpoc
from sklearn.metrics.cluster import  mutual_info_score

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

def add_gaussian_noise():
    mean = 0
    std_dev = 1
    noise = np.random.normal(mean, std_dev)
    noise = np.clip(noise, -3, 3)  # 将噪声限制在正负3以内的范围内
    return noise

def generate_overlap_coordinates(image_width, image_height, subimage_width, subimage_height, overlap_threshold):
    # 计算可允许的最大偏移量
    max_offset_x = subimage_width - int(subimage_width * overlap_threshold)
    max_offset_y = subimage_height - int(subimage_height * overlap_threshold)

    # 随机生成第一个子图像的左上角坐标
    x1 = np.random.randint(0, image_width - subimage_width + 1)
    y1 = np.random.randint(0, image_height - subimage_height + 1)

    # 随机生成第二个子图像的左上角坐标，确保重叠度不低于阈值
    x2 = np.random.randint(max(0, x1 - max_offset_x), min(image_width - subimage_width, x1 + max_offset_x) + 1)
    y2 = np.random.randint(max(0, y1 - max_offset_y), min(image_height - subimage_height, y1 + max_offset_y) + 1)

    return (x1, y1), (x2, y2)


# 大图像尺寸
image_width = 1080
image_height = 613

# 子图像尺寸
subimage_width = 100
subimage_height = 100

# 重叠度阈值
overlap_threshold = 0.5

for i in range(1,100):
    subimage1_corner, subimage2_corner = generate_overlap_coordinates(image_width, image_height, subimage_width,
                                                                      subimage_height, overlap_threshold)
    sub1_end = tuple(x + 100 for x in subimage1_corner)
    sub2_end = tuple(x + 100 for x in subimage2_corner)
    img1, img2, img3 = image_process.process_image('data/map1.jpeg','','','',subimage1_corner,sub1_end,subimage2_corner,sub2_end,False)
    # 生成两个子图像的左上角坐标
    # plt.figure()
    # plt.imshow(img1, vmin=img1.min(), vmax=img1.max(), cmap='gray')
    # plt.show()
    # plt.figure()
    # plt.imshow(img2, vmin=img2.min(), vmax=img2.max(), cmap='gray')
    # plt.show()
    # plt.figure()
    # plt.imshow(img3, vmin=img3.min(), vmax=img3.max(), cmap='gray')
    # plt.show()
    cv2.imwrite("data/random_1.jpg", img3)
    matcher= imregpoc.imregpoc(img1,img2)
    img4 = matcher.stitching()
    angle = random.uniform(-10, 10)
    img1 = rotate_image(img1, angle)
    img2 = rotate_image(img2, angle)
    # plt.figure()
    # plt.imshow(img1, vmin=img1.min(), vmax=img1.max(), cmap='gray')
    # plt.show()
    # plt.figure()
    # plt.imshow(img2, vmin=img2.min(), vmax=img2.max(), cmap='gray')
    # plt.show()
    height, width = img3.shape
    temp = img3.copy()
    hei, wid = img4.shape
    h = min(hei, height)
    w = min(wid, width)
    img4 = img4[:h, :w]
    img3 = img3[:h, :w]
    cv2.imwrite("data/random_2.jpg", img4)

    blur_coordinate1 = round(subimage1_corner[0] + add_gaussian_noise()), round(
        subimage1_corner[1] + add_gaussian_noise())
    blur_coordinate2 = round(subimage2_corner[0] + add_gaussian_noise()), round(
        subimage2_corner[1] + add_gaussian_noise())
    img1_r = rotate_image(img1, -angle/2)
    # plt.figure()
    # plt.imshow(img1_r, vmin=img1_r.min(), vmax=img1_r.max(), cmap='gray')
    # plt.show()
    img2_r = rotate_image(img2, -angle/2)
    # plt.figure()
    # plt.imshow(img2_r, vmin=img2_r.min(), vmax=img2_r.max(), cmap='gray')
    # plt.show()

    mse = judge.MSE_calculate(img3,img4)
    img3_l = img3.reshape(-1)
    img4_l = img4.reshape(-1)
    MIValue = mutual_info_score(img3_l, img4_l)

    img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2RGB)
    img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2RGB)
    ssim = SSIM.calculate_ssim(img3,img4,4)
    psnr = PSNR.calculate_psnr(img3,img4,4)

    img_dir = image_process.merge_images(img1_r, img2_r, blur_coordinate1, blur_coordinate2)
    hei, wid = img_dir.shape
    h = min(hei, height)
    w = min(wid, width)
    img_dir = img_dir[:h, :w]
    img3 = temp[:h, :w]
    mse_r = judge.MSE_calculate(img3, img_dir)
    img_dir_l = img_dir.reshape(-1)
    img3_l = img3.reshape(-1)
    MIValue_r = mutual_info_score(img3_l, img_dir_l)
    img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2RGB)
    img_dir = cv2.cvtColor(img_dir, cv2.COLOR_GRAY2RGB)
    ssim_r = SSIM.calculate_ssim(img3, img_dir, 4)
    psnr_r = PSNR.calculate_psnr(img3, img_dir, 4)
    with open(f"data/mse.txt", "a") as file:
        file.write(f"{mse} {mse_r}\n")
    with open(f"data/mi.txt", "a") as file:
        file.write(f"{MIValue} {MIValue_r}\n")
    with open(f"data/ssim.txt", "a") as file:
        file.write(f"{ssim} {ssim_r}\n")
    with open(f"data/psnr.txt", "a") as file:
        file.write(f"{psnr} {psnr_r}\n")







print("第一个子图像左上角坐标:", subimage1_corner)
print("第二个子图像左上角坐标:", subimage2_corner)