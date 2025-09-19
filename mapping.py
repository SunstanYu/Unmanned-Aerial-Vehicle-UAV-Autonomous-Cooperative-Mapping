import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
import fusion
import imregpoc


class Map:
    def __init__(self, map_current, size=(0, 0)):
        self.map: MapTemp = map_current
        self.map_current: MapTemp = map_current
        self.transform_matrix = None

    def border(self, img):
        image = img.copy()
        edge_threshold = 50
        edge_mask = (image < edge_threshold)

        # 使用距离变换来找到边缘区域中每个像素点到最近非边缘像素点的距离
        dist_transform = cv2.distanceTransform((~edge_mask).astype(np.uint8), cv2.DIST_L2, 3)
        dist_transform[edge_mask] += 10
        # 创建一个空的图像用于存储结果
        result = np.zeros_like(image)

        # 对于边缘区域的每个像素点，找到最近邻居像素点的索引
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if edge_mask[y, x]:  # 如果是边缘像素点
                    # 在一个小的邻域内寻找最近邻像素点
                    neighborhood_size = 3  # 可以根据实际情况调整
                    y_start = max(0, y - neighborhood_size)
                    y_end = min(image.shape[0], y + neighborhood_size)
                    x_start = max(0, x - neighborhood_size)
                    x_end = min(image.shape[1], x + neighborhood_size)

                    # 在邻域内寻找最近邻像素点的索引
                    neighborhood = dist_transform[y_start:y_end, x_start:x_end]
                    nearest_indices = np.where(neighborhood == np.min(neighborhood))
                    nearest_y, nearest_x = nearest_indices[0][0] + y_start, nearest_indices[1][0] + x_start

                    # 使用最近邻像素点的像素值替换当前像素点的像素值
                    result[y, x] = image[nearest_y, nearest_x]
                else:
                    result[y, x] = image[y, x]

        return result

    def update(self, map_current):
        # t = self.t - self.t0
        # dx = self.vx * t
        # dy = self.vy * t
        self.map_current = map_current
        # plt.figure()
        # plt.imshow(self.map_current.map, vmin=0, vmax=255, cmap='gray')
        # plt.imsave("data/mappiece/before_border.png", self.map_current.map)
        # plt.show()
        self.map_current.map = self.border(self.map_current.map)
        # plt.figure()
        # plt.imshow(self.map_current.map, vmin=0, vmax=255, cmap='gray')
        # plt.imsave("data/mappiece/after_border.png", self.map_current.map)
        # plt.show()
        ori_center = [self.map_current.x + self.map_current.map.shape[1] // 2,
                      self.map_current.y + self.map_current.map.shape[0] // 2]
        translation_x = abs(self.map.x - ori_center[0])
        translation_y = abs(self.map.y - ori_center[1])
        center = (self.map_current.map.shape[1] // 2, self.map_current.map.shape[0] // 2)
        rotation = self.map_current.r
        self.transform_matrix = cv2.getRotationMatrix2D(center, rotation, 1)
        xmax, xmin, ymax, ymin = self.convertRectangle(self.transform_matrix)
        height, width = self.map_current.map.shape
        height_map, width_map = self.map.map.shape
        sxmax = max(xmax, width - 1)
        sxmin = min(xmin, 0)
        symax = max(ymax, height - 1)
        symin = min(ymin, 0)
        swidth, sheight = sxmax - sxmin + 1, symax - symin + 1
        coordinates = (sxmax, sxmin, symax, symin)
        new_ori = (
            round(self.map_current.x + 0.5 * width - swidth // 2),
            round(self.map_current.y + 0.5 * height - sheight // 2))
        related_ori = (new_ori[0] - self.map.x, new_ori[1] - self.map.y)
        related_center = (related_ori[0] + swidth // 2, related_ori[1] + sheight // 2)
        # related_ori：变换后相对起始坐标； related_center：变换后相对中心坐标
        match_cmp, match_x, match_y = self.cornel_detection((sheight, swidth), related_center)
        if not np.any(match_cmp):
            self.map_no_cover(translation_x, translation_y, coordinates, new_ori, ori_center)
        else:
            self.map_cover(match_cmp, match_x, match_y, translation_x, translation_y, coordinates, new_ori)
        return

    def convertRectangle(self, rotation_matrix):
        height, width = self.map_current.map.shape
        rectangles = np.float32([0, 0, 1, width - 1, 0, 1, 0, height - 1, 1, width - 1, height - 1, 1]).reshape(4, 3)
        transformed_points = np.dot(rectangles, rotation_matrix.T)
        xmax = math.ceil(transformed_points[:, 0].max())
        xmin = math.floor(transformed_points[:, 0].min())
        ymax = math.ceil(transformed_points[:, 1].max())
        ymin = math.floor(transformed_points[:, 1].min())
        return [xmax, xmin, ymax, ymin]

    def map_no_cover(self, translation_x, translation_y, coordinates, new_ori, ori_center):
        height, width = self.map.map.shape
        sxmax, sxmin, symax, symin = coordinates
        swidth, sheight = sxmax - sxmin + 1, symax - symin + 1
        xtrans, ytrans = 0 - sxmin, 0 - symin
        # print(f"xtrans:{xtrans},ytrans:{ytrans}")
        print("map no cover")
        # new_ori = (self.map_current.x + 0.5 * width - swidth // 2, self.map_current.y + 0.5 * height - sheight // 2)

        # 旋转轴为z轴，起点x轴, r [-180, 180]
        # map_origin: 地图左上角起点坐标 new_ori: 变换后起点坐标
        if ori_center[0] <= self.map.x and ori_center[1] <= self.map.y:
            self.transform_matrix[:, 2] += [xtrans, ytrans]
            map_origin = (xtrans + swidth // 2 + translation_x, ytrans + sheight // 2 + translation_y)
            map_current_transformed = cv2.warpAffine(self.map_current.map, self.transform_matrix,
                                                     (map_origin[0] + width,
                                                      map_origin[1] + height))
            black_pixel = self.map.map <= 50
            map_current_transformed[map_origin[1]:map_origin[1] + height,
            map_origin[0]:map_origin[0] + width][~black_pixel] = self.map.map[~black_pixel]
            self.map.x = new_ori[0]
            self.map.y = new_ori[1]
            self.map.map = map_current_transformed

        elif ori_center[0] <= self.map.x and ori_center[1] > self.map.y:
            if translation_y - sheight // 2 < 0:
                self.transform_matrix[:, 2] += [xtrans, ytrans]
                map_origin = (swidth // 2 + translation_x, ytrans + sheight // 2 - translation_y)
            else:
                self.transform_matrix[:, 2] += [xtrans, ytrans + translation_y - sheight // 2]
                map_origin = (swidth // 2 + translation_x, 0)
            map_current_transformed = cv2.warpAffine(self.map_current.map, self.transform_matrix,
                                                     (map_origin[0] + width,
                                                      max(map_origin[1] + height,
                                                          translation_y + sheight // 2)))
            black_pixel = self.map.map <= 50
            map_current_transformed[map_origin[1]:map_origin[1] + height,
            map_origin[0]: map_origin[0] + width][~black_pixel] = self.map.map[~black_pixel]
            self.map.x = new_ori[0]
            self.map.y = map_origin[1] + self.map.y
            self.map.map = map_current_transformed

        elif ori_center[0] > self.map.x and ori_center[1] <= self.map.y:
            if translation_x - swidth // 2 < 0:
                self.transform_matrix[:, 2] += [xtrans, ytrans]
                map_origin = (xtrans + swidth // 2 - translation_x, sheight // 2 + translation_y)
            else:
                self.transform_matrix[:, 2] += [xtrans + translation_x - swidth // 2, ytrans]
                map_origin = (0, sheight // 2 + translation_y)
            map_current_transformed = cv2.warpAffine(self.map_current.map, self.transform_matrix,
                                                     (max(map_origin[0] + width,
                                                          translation_x + swidth // 2),
                                                      map_origin[1] + height))
            black_pixel = self.map.map <= 50
            map_current_transformed[map_origin[1]:map_origin[1] + height,
            map_origin[0]:map_origin[0] + width][~black_pixel] = self.map.map[~black_pixel]
            self.map.x = map_origin[0] + self.map.x
            self.map.y = new_ori[1]
            self.map.map = map_current_transformed

        elif ori_center[0] > self.map.x and ori_center[1] > self.map.y:
            if translation_x - swidth // 2 < 0 and translation_y - sheight // 2 < 0:
                self.transform_matrix[:, 2] += [xtrans, ytrans]
                map_origin = (xtrans + swidth // 2 - translation_x, ytrans + sheight // 2 - translation_y)
                self.map.x = new_ori[0]
                self.map.y = new_ori[1]
            elif translation_x - swidth // 2 < 0 and translation_y - sheight // 2 > 0:
                self.transform_matrix[:, 2] += [xtrans, ytrans + translation_y - sheight // 2]
                map_origin = (xtrans + swidth // 2 - translation_x, 0)
                self.map.x = new_ori[0]
                self.map.y = map_origin[1] + self.map.y
            elif translation_x - swidth // 2 > 0 and translation_y - sheight // 2 < 0:
                self.transform_matrix[:, 2] += [xtrans + translation_x - swidth // 2, ytrans]
                map_origin = (0, ytrans + sheight // 2 - translation_y)
                self.map.x = map_origin[0] + self.map.x
                self.map.y = new_ori[1]
            else:
                self.transform_matrix[:, 2] += [xtrans + translation_x - swidth // 2,
                                                ytrans + translation_y - sheight // 2]
                map_origin = (0, 0)
            # 98 102
            map_current_transformed = cv2.warpAffine(self.map_current.map, self.transform_matrix,
                                                     (max(map_origin[0] + width,
                                                          translation_x + swidth // 2),
                                                      max(map_origin[1] + height,
                                                          translation_y + sheight // 2)))
            # plt.figure()
            # plt.imshow(map_current_transformed, vmin=map_current_transformed.min(), vmax=map_current_transformed.max(), cmap='gray')
            # plt.show()
            black_pixel = map_current_transformed <= 50
            black_pixel = black_pixel[map_origin[1]:map_origin[1] + height,
            map_origin[0]:map_origin[0] + width]
            map_current_transformed[map_origin[1]:map_origin[1] + height,
            map_origin[0]:map_origin[0] + width][black_pixel]  = self.map.map[black_pixel]
            # black_pixel = self.map.map != 0
            # map_current_transformed[map_origin[1]:map_origin[1] + height,
            # map_origin[0]:map_origin[0] + width][black_pixel]  = self.map.map[black_pixel]
            self.map.map = map_current_transformed

        # else:
        #     self.transform_matrix[:, 2] += [xtrans + translation_x, ytrans + translation_y]
        #     map_current_transformed = cv2.warpAffine(self.map_current.map, self.transform_matrix,
        #                                              (xtrans + swidth, ytrans + sheight))
        #     black_pixel = map_current_transformed == 0
        #     self.map.map[translation_y - sheight // 2:translation_y + sheight // 2,
        #     translation_x - swidth // 2:translation_x + swidth // 2][~black_pixel] = map_current_transformed[
        #         ~black_pixel]
        #     self.map.map = map_current_transformed

        cv2.imwrite("data/bigmap.jpg", self.map.map)

    def cornel_detection(self, size, coordination):
        x, y = coordination
        # new map center in the origin map
        height, width = size
        step = 1
        threshold_x = round(width / 0.8)
        threshold_y = round(height / 0.8)
        # 0.8 scale 0.64 cover
        height_map, width_map = self.map.map.shape
        left_side = x - threshold_x // 2
        right_side = x + threshold_x // 2
        up_side = y - threshold_y // 2
        down_side = y + threshold_y // 2
        if 0 < left_side < width_map or 0 < right_side < width_map or (left_side<0 and right_side>width_map):
            if 0 < up_side < height_map or 0 < down_side < height_map or (up_side<0 and down_side>height_map):
                for i in range(max(left_side, 0), min(right_side, width_map), step):
                    for j in range(max(up_side, 0), min(down_side, height_map), step):
                        map_final = np.zeros(size)
                        if i + width <= width_map and j + height <= height_map:
                            map_tmp = self.map.map[j:min(j + height, down_side), i: min(i + width, right_side)]
                            hei, wid = map_tmp.shape
                            map_final[:hei, :wid] = map_tmp
                            num_black = np.sum(map_final != 0)
                            if num_black >= math.ceil(height * width * 0.55):
                                return map_final, i, j
                        else:
                            x_rest = width_map - i
                            y_rest = height_map - j
                            map_tmp = self.map.map[j:min(j + y_rest, down_side, j + height), i:min(i + x_rest, right_side, i + width)]
                            print(f"map_tmp size: {map_tmp.shape}; i: {i}; j: {j}; y_rest: {y_rest}; x_rest: {x_rest}; downside:{down_side}; right_side:{right_side}")
                            hei, wid = map_tmp.shape
                            map_final[:hei, :wid] = map_tmp
                            num_black = np.sum(map_final != 0)
                            if num_black >= math.ceil(height * width * 0.55):
                                return map_final, i, j
                                # start_x = i + x_rest - width
                                # start_y = j + y_rest - height
                                # map_out = self.map.map[start_y:start_y + height, start_x:start_x + width]
                                # return map_out, start_x, start_y

        return None, 0, 0

    def map_cover(self, match_cmp, match_x, match_y, translation_x, translation_y, coordinates, new_ori):
        # plt.figure()
        # plt.imshow(match_cmp, vmin=match_cmp.min(), vmax=match_cmp.max(), cmap='gray')
        # plt.show()
        print("map cover")
        height, width = self.map.map.shape
        sxmax, sxmin, symax, symin = coordinates
        swidth, sheight = sxmax - sxmin + 1, symax - symin + 1
        xtrans, ytrans = 0 - sxmin, 0 - symin
        self.transform_matrix[:, 2] += [xtrans, ytrans]
        map_current_transformed = cv2.warpAffine(self.map_current.map, self.transform_matrix,
                                                 (swidth, sheight))
        padding = 2
        # 1st transform
        result = imregpoc.imregpoc(match_cmp, map_current_transformed)
        now_xmin, now_ymin, now_xmax, now_ymax = result.convertRectangle()
        now_width, now_height = now_xmax - now_xmin + 1, now_ymax - now_ymin + 1
        now_ori = (match_x + now_xmin, match_y + now_ymin)
        # 相对匹配点的坐标
        sub_map, sub_width, sub_height, sub_xtrans, sub_ytrans = result.stitching()
        # new_ori = (self.map_current.x + 0.5 * width - swidth // 2, self.map_current.y + 0.5 * height - sheight // 2)

        # 旋转轴为z轴，起点x轴, r [-180, 180]
        # ori_center: 地图左上角起点坐标 new_ori: 变换后起点坐标
        if now_ori[0] <= 0 and now_ori[1] <= 0:
            # 地图左上角
            new_map = np.zeros((sub_ytrans + max(height, 50) - match_y, sub_xtrans + max(width, 50) - match_x), np.float32)
            # 假设这种情况拼接图像超出左上角
            new_map[sub_ytrans - match_y:sub_ytrans - match_y + height,
            sub_xtrans - match_x:sub_xtrans - match_x + width] = self.map.map
            sub_map = sub_map[:sub_height-padding, :sub_width-padding]
            black_pixel = sub_map <= 50
            new_map[0:sub_height-padding, 0:sub_width-padding][~black_pixel] = sub_map[~black_pixel]
            self.map.x += now_ori[0]
            self.map.y += now_ori[1]
            self.map.map = new_map

        elif now_ori[0] <= 0 < now_ori[1]:
            # 地图左侧
            new_map = np.zeros((max(sub_height - sub_ytrans + match_y, max(height,50)), sub_xtrans + max(width, 50) - match_x))
            new_map[0:height, sub_xtrans - match_x:sub_xtrans - match_x + width] = self.map.map
            sub_map = sub_map[padding:sub_height-padding, padding:sub_width-padding]
            black_pixel = sub_map <= 50
            new_map[match_y - sub_ytrans+padding:match_y - sub_ytrans + sub_height-padding, padding:sub_width-padding][~black_pixel] = sub_map[
                ~black_pixel]
            self.map.x += now_ori[0]
            self.map.y = self.map.y
            self.map.map = new_map

        elif now_ori[0] > 0 >= now_ori[1]:
            # 地图上侧
            new_map = np.zeros((sub_ytrans + max(height,50) - match_y, max(sub_width - sub_xtrans + match_x, max(width, 50))))
            new_map[sub_ytrans - match_y:sub_ytrans - match_y + height, 0:width] = self.map.map
            sub_map = sub_map[padding:sub_height-padding, padding:sub_width-padding]
            black_pixel = sub_map <= 50
            new_map[padding:sub_height-padding, match_x - sub_xtrans+padding:match_x - sub_xtrans + sub_width-padding][~black_pixel] = sub_map[
                ~black_pixel]
            self.map.x = self.map.x
            self.map.y += now_ori[1]
            self.map.map = new_map

        elif now_ori[0] + sub_width > width or now_ori[1] + sub_height > height:
            # 地图右下角
            new_map = np.zeros((max(sub_height - sub_ytrans + match_y, height), max(sub_width - sub_xtrans + match_x, width)))
            new_map[0:height, 0:width] = self.map.map
            sub_map = sub_map[padding:sub_height-padding, padding:sub_width-padding]
            black_pixel = sub_map <= 50
            new_map[match_y - sub_ytrans + padding:match_y - sub_ytrans + sub_height - padding,
            match_x - sub_xtrans + padding:match_x - sub_xtrans + sub_width - padding][~black_pixel] = sub_map[~black_pixel]
            self.map.map = new_map

        else:
            # 地图内部

            sub_map = sub_map[padding:sub_height-padding, padding:sub_width-padding]
            black_pixel = sub_map <= 50
            print(f"black_pixel size: {black_pixel.shape}; map size:{self.map.map.shape}; submap size: {sub_map.shape}")
            print(f"match_x: {match_x}; match_y: {match_y}: sub_xtrans: {sub_xtrans};sub_ytrans: {sub_ytrans}")
            print(f"sub_width: {sub_width}; sub_height:{sub_height}")
            self.map.map[match_y - sub_ytrans+padding:match_y - sub_ytrans + sub_height-padding,
            match_x - sub_xtrans+padding:match_x - sub_xtrans + sub_width-padding][~black_pixel] = sub_map[~black_pixel]
            # print(f"black_pixel size: {black_pixel.shape}; map size:{self.map.map.shape}; submap size: {sub_map.shape}")



# def image_transform(self, map_process):
#     pitch_angle = map_process.r[0]  # 俯仰角（度）
#     roll_angle = map_process.r[1]  # 滚转角（度）
#     yaw_angle = map_process.r[2]  # 偏航角（度）
#     M = cv2.getRotationMatrix2D((map_process.map.shape[1] // 2, map_process.map.shape[0] // 2), roll_angle, 1)
#
#     # 应用仿射变换
#     transformed_image = cv2.warpAffine(map_process.map, M, (map_process.map.shape[1], map_process.map.shape[0]))


'''
x,y 是左上角坐标
'''


class MapTemp:
    def __init__(self, map, x, y, r=0):
        self.map = map.astype(np.float32)
        self.x = x
        self.y = y
        self.r = r


# i1 = cv2.imread("data/map1_1.jpg")
# i2 = cv2.imread("data/map1_2.jpg")
#
# map1 = MapTemp(cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY), 35, 35, (0, 0, 0))
# map2 = MapTemp(cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY), 45, 45, (0, 0, 0))
# mapping = Map(map_current=map1)
# mapping.update(map2, 1, 1)
# plt.figure()
# plt.imshow(mapping.map.map, vmin=mapping.map.map.min(), vmax=mapping.map.map.max(), cmap='gray')
# plt.show()
# # cv2.imshow("Transformed Image", mapping.map.map)
# print(f"New origin: {mapping.map.x}, {mapping.map.y}")
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
