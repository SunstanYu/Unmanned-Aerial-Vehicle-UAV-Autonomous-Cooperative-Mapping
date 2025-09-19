import cv2
from PIL import Image
import numpy as np

import display


def downsample_image(input_path, output_path, factor):
    # 打开图像文件
    original_image = Image.open(input_path)

    # 将图像转换为灰度图像
    gray_image = original_image.convert('L')

    # 将 PIL 图像转换为 NumPy 数组
    image_array = np.array(gray_image)

    # 下采样
    downsampled_array = image_array[::factor, ::factor]

    # 从 NumPy 数组创建新的 PIL 图像对象
    downsampled_image = Image.fromarray(downsampled_array)

    # 将 PIL 图像对象保存为 JPEG 文件
    downsampled_image.save(output_path)

    print("下采样处理完成，保存为", output_path)

def process_image(input_path, output_path1,output_path2,output_path3, start1, end1,start2,end2, isSave):
    # 打开图像文件
    original_image = Image.open(input_path)

    # 将图像转换为灰度图像
    gray_image = original_image.convert('L')

    # 获取左上角32x32像素的图像
    cropped_image1 = gray_image.crop((start1[0], start1[1], end1[0], end1[1]))

    # 将 PIL 图像对象保存为 JPEG 文件

    # 获取左上角32x32像素的图像
    cropped_image2 = gray_image.crop((start2[0], start2[1], end2[0], end2[1]))

    img = merge_images(np.array(cropped_image1),np.array(cropped_image2),start1,start2)
    if isSave:
        cv2.imwrite(output_path3, img)
        cropped_image1.save(output_path1)
        # 将 PIL 图像对象保存为 JPEG 文件
        cropped_image2.save(output_path2)
    return np.array(cropped_image1),np.array(cropped_image2),img

    # print("处理完成，保存为", output_path)


def merge_images(image1, image2, position1, position2):
    # 计算合并后图像的尺寸
    min_x = min(position1[0], position2[0])
    min_y = min(position1[1], position2[1])
    max_x = max(position1[0] + image1.shape[1], position2[0] + image2.shape[1])
    max_y = max(position1[1] + image1.shape[0], position2[1] + image2.shape[0])
    width = max_x - min_x
    height = max_y - min_y

    # 创建合并后图像
    merged_image = np.zeros((height, width), dtype=np.uint8)

    # 将图像1和图像2放置在合并图像中的对应位置
    merged_image[position1[1] - min_y:position1[1] - min_y + image1.shape[0],
    position1[0] - min_x:position1[0] - min_x + image1.shape[1]] = image1
    merged_image[position2[1] - min_y:position2[1] - min_y + image2.shape[0],
    position2[0] - min_x:position2[0] - min_x + image2.shape[1]] = image2

    return merged_image

def draw_points_and_lines(image, coordinates, c=None):
    coordinates = np.array(coordinates).astype(int)
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算图像底边中点的坐标
    mid_x, mid_y = int(width / 2), height - 1

    # 循环遍历坐标列表并在图像上绘制点和线
    if [c]:
        cv2.putText(image, f'({c[0]},{c[1]},{c[2]})', (mid_x - 30, mid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0),
                    1)
    for x, y in coordinates:
        # 绘制点
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        # 绘制线
        cv2.line(image, (x, y), (mid_x, mid_y), (0, 255, 0), 2)

    # 返回绘制完成的图像
    return image


# img_path = "data/map1.jpeg"
# img_downsample_path="data/map1_down.jpg"
# out1_path = "data/map1_1.jpg"
# out2_path = "data/map1_2.jpg"  # 替换为你的输出图像路径
# out3_path = "data/map1_3.jpg"
# start1 = 35
# start3 = 35
# start2 = 45
# start4 = 45
# #horizonal: 1. 0.56 cover 2. 0.53 3. 0.59 4. 0.56
# #vertical: 1. 0.56 cover  2. 0.53 3. 0.53 4. 0.53
# #diaganal: 1. 0.39 cover  2. 0.47 3. 0.47 4. 0.51/ 0.35/0.51
# downsample_image(img_path,img_downsample_path,5)
# process_image(img_downsample_path, out1_path, out2_path,out3_path,[start1, start3], [start1+32, start3+32],[start2, start4], [start2+32, start4+32])
#
# image1 = cv2.imread(out1_path, cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread(out2_path, cv2.IMREAD_GRAYSCALE)

# fast = cv2.FastFeatureDetector_create()
# keypoints1 = fast.detect(image1, None)
# keypoints2 = fast.detect(image2, None)
#
# final_img1 = draw_points_and_lines(image1,keypoints1)
# display.show_img(final_img1)
