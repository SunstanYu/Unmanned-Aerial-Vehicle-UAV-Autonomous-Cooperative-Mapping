import math

import cv2
import numpy as np
import imregpoc
import judge
from sklearn.metrics.cluster import  mutual_info_score
import matplotlib.pyplot as plt

def trans(original_image, angle):
    height, width = original_image.shape
    M = cv2.getRotationMatrix2D((original_image.shape[1] // 2, original_image.shape[0] // 2), angle, 1)
    rectangles = np.float32([0, 0, 1, width - 1, 0, 1, 0, height - 1, 1, width - 1, height - 1, 1]).reshape(4, 3)
    transformed_points = np.dot(rectangles, M.T)
    tx = 0
    ty = 0
    xmax = math.ceil(transformed_points[:, 0].max())
    xmin = math.floor(transformed_points[:, 0].min())
    ymax = math.ceil(transformed_points[:, 1].max())
    ymin = math.floor(transformed_points[:, 1].min())
    sxmax = max(xmax, width - 1)
    sxmin = min(xmin, 0)
    symax = max(ymax, height - 1)
    symin = min(ymin, 0)
    swidth, sheight = sxmax - sxmin + 1, symax - symin + 1
    xtrans, ytrans = 0 - sxmin, 0 - symin
    # 将结果的齐次坐标转换回二维坐标形式
    M[:, 2] += [xtrans, ytrans]
    print(swidth + tx, sheight + ty)
    # 应用仿射变换
    transformed_image = cv2.warpAffine(original_image, M, (swidth + tx, sheight + ty))
    return transformed_image


#测试其他
# def count_zeros(matrix):
#     # 将矩阵中等于 0 的元素转换为布尔值，True 表示为 0，False 表示不为 0
#     # zero_mask = (matrix == 0)
#     # 使用 np.sum 函数计算布尔值矩阵中 True 的数量，即为值为 0 的元素数量
#     zero_count = np.sum(matrix == 0)
#     return zero_count
#
# # 测试代码
# matrix = np.array([[1, 0, 3],
#                    [4, 0, 6],
#                    [0, 8, 9]])
#
# zeros = count_zeros(matrix)
# print("值为 0 的元素数量为:", zeros)
# def rotate_image(image, angle):
#     center = (image.shape[1] // 2, image.shape[0] // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
#     return rotated_image
#
# # 读取图像
# image = cv2.imread("data/map1_1.jpg")
#
# # 旋转图像
# angle = 30
# rotated_image = rotate_image(image, angle)
# plt.figure()
# plt.imshow(rotated_image, vmin=rotated_image.min(), vmax=rotated_image.max(), cmap='gray')
# plt.show()
# # 再次旋转图像
# rotated_back_image = rotate_image(rotated_image, -15)
# plt.figure()
# plt.imshow(rotated_back_image, vmin=rotated_back_image.min(), vmax=rotated_back_image.max(), cmap='gray')
# plt.show()

def add_gaussian_noise(image, mean=10, std=10):
    """
    给图像添加高斯噪声。

    参数：
        image: 输入的图像（灰度图像）
        mean: 高斯噪声的均值，默认为0
        std: 高斯噪声的标准差，默认为25
    返回：
        添加了高斯噪声的图像
    """
    # 生成符合高斯分布的随机数，均值为mean，标准差为std
    # 生成符合高斯分布的随机数，均值为mean，标准差为std
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)

    # 裁剪噪声值到指定范围
    noise = np.clip(noise, 0, 80)    # noise = np.clip(noise, -20, 20)
    # 将随机数添加到图像的像素值中
    noisy_image = cv2.add(image, noise)
    return noisy_image

img = cv2.imread("data/yongzhou-small/img_w.jpg", 0)
white = img==255
img_b = img.copy()
img_b[white] = 0
cv2.imwrite("data/yongzhou-small/img_wb.jpg",img_b)
# 读取图像
# image = cv2.imread("data/random_1.jpg", cv2.IMREAD_GRAYSCALE)
#
# # 添加高斯噪声
# noisy_image = add_gaussian_noise(image)
#
# # 显示原始图像和添加了高斯噪声的图像
# cv2.imshow("Original Image", image)
# cv2.imshow("Noisy Image", noisy_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#测试互信息
# img1 = cv2.imread("data/random_2.jpg")
# img2 = cv2.imread("data/random_1.jpg")
# img1_l = img1.reshape(-1)
# img2_l = img2.reshape(-1)
# MIValue = mutual_info_score(img1_l,img2_l)
# print(MIValue)

# 测试拼接
# ref: np.ndarray = cv2.imread('data/yongzhou-small/DSC03198.JPG', 0)
# cmp: np.ndarray = cv2.imread('data/yongzhou-small/DSC03199.JPG', 0)
# # ref: np.ndarray = cv2.imread('data/heatmap95.jpg', 0)
# # cmp: np.ndarray = cv2.imread('data/heatmap100.jpg', 0)
# # tran_cmp = trans(cmp, 10)
# height, width = ref.shape
# # ref_center = [ref.shape[1] // 2, ref.shape[0] // 2]
# # ref_cp = ref.copy()
# # ref_cp[ref_center[0] - 1:ref_center[0] + 1, :] = 0
# # ref_cp[:, ref_center[1] - 1:ref_center[1] + 1] = 0
#
# # M = cv2.getRotationMatrix2D((cmp.shape[1] // 2, cmp.shape[0] // 2), 90, 1)
# # #
# # # # 应用仿射变换
# # transformed_image = cv2.warpAffine(cmp, M, (cmp.shape[1], cmp.shape[0]))
# # cv2.imshow("Transformed Image", ref_cp)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # print(transformed_image.shape)
# # temp=ref[height-32:, width-32:]
# matcher = imregpoc.imregpoc(ref, cmp)
# # matcher = imregpoc.TempMatcher(ref,'SIFT')
# # matcher.match(cmp,1)
# # center = np.array(ref.shape) / 2
# # pers = imregpoc.imregpoc.poc2warp(center, [10, 10, 0, 0])
# print(matcher.getPerspective())
# # print(matcher.isSucceed())
# img = matcher.stitching()
# # out = "data/map1_4.jpg"
# cv2.imwrite("data/test_new1.jpg", ref)
# cv2.imwrite("data/test_new2.jpg", cmp)
# cv2.imwrite("data/test_new3.jpg", img)
# jud: np.ndarray = cv2.imread('data/map1_3.jpg', 0)
# # jud = np.zeros(img.shape)
# judge.MSE_calculate(jud,img)
# result.showLPA()
# result.showLPB()
# result.showRotatePeak()
# result.showTranslationPeak()

# 测试变换
# original_image = cv2.imread("data/map1_1.jpg")
# original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
# # 无人机位姿（示例值）
# pitch_angle = 10  # 俯仰角（度）
# roll_angle = 100    # 滚转角（度）
# yaw_angle = 0     # 偏航角（度）
# tx = 0
# ty = 0
# # 计算仿射变换矩阵
# # 这里使用示例的变换方式，你需要根据实际情况计算
# height,width = original_image.shape
# M = cv2.getRotationMatrix2D((original_image.shape[1] // 2, original_image.shape[0] // 2), roll_angle, 1)
# rectangles = np.float32([0,0,1, width-1,0,1, 0,height-1,1, width-1, height-1,1]).reshape(4,3)
# transformed_points = np.dot(rectangles, M.T)
# xmax = math.ceil(transformed_points[:, 0].max())
# xmin = math.floor(transformed_points[:, 0].min())
# ymax = math.ceil(transformed_points[:, 1].max())
# ymin = math.floor(transformed_points[:, 1].min())
# sxmax = max(xmax, width - 1)
# sxmin = min(xmin, 0)
# symax = max(ymax, height - 1)
# symin = min(ymin, 0)
# swidth, sheight = sxmax - sxmin + 1, symax - symin + 1
# xtrans, ytrans = 0 - sxmin, 0 - symin
# # 将结果的齐次坐标转换回二维坐标形式
# M[:, 2] += [xtrans, ytrans]
# # 应用仿射变换
# transformed_image = cv2.warpAffine(original_image, M, (swidth+tx+100,sheight+ty+100))
# print(transformed_image.shape)
# # 显示结果
# # background = np.full((100,100), 255).astype(np.uint8)
# # black_pixel = transformed_image == 0
# # background[0:sheight,0:swidth][~black_pixel]= transformed_image[~black_pixel]
# # a= np.array([1,2,3],[4,5,6])
# cv2.imshow("Transformed Image", transformed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


