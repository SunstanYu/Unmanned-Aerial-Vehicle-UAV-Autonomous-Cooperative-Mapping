import mapping
import cv2
from matplotlib import pyplot as plt

# 调用函数并传入文件路径
# file_path = "position.txt"  # 替换为你的文件路径
# with open(file_path, 'r') as file:
#     # 读取文件内容
#     content = file.read()
#     lines = content.split('\n')
#     num = 0
#     for line in lines[3:]:
#         coordinates = line.split(' ')
#         print(coordinates)
#         x = int(float(coordinates[1])/10)
#         y = int(float(coordinates[2])/10)
#         z = int(float(coordinates[3])/10)
#         img = cv2.imread(f"data/heatmap/heatmap{num}.jpg")
#         image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         map_temp = mapping.MapTemp(image, x, y)
#         if num == 10:
#             print("test")
#         if num == 0:
#             ground = mapping.Map(map_current=map_temp)
#         else:
#             ground.update(map_temp)
#         num += 5
#         plt.figure()
#         plt.imshow(ground.map.map, vmin=ground.map.map.min(), vmax=ground.map.map.max(), cmap='gray')
#         plt.show()


# 调用函数并传入文件路径
file_path = "data/mappiece/labels.txt"  # 替换为你的文件路径
with open(file_path, 'r') as file:
    # 读取文件内容
    content = file.read()
    lines = content.split('\n')
    num = 0
    for line in lines:
        coordinates = line.split(',')
        print(coordinates)
        x = int(float(coordinates[0]))
        y = int(float(coordinates[1]))
        r = 0
        img = cv2.imread(f"data/mappiece/img{num}.png")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        map_temp = mapping.MapTemp(image, x, y, r)
        # if num == 5:
        #     print("test")
        if num == 0:
            ground = mapping.Map(map_current=map_temp)
        else:
            ground.update(map_temp)
        num += 1
        # plt.figure()
        # plt.imshow(ground.map.map, vmin=ground.map.map.min(), vmax=ground.map.map.max(), cmap='gray')
        # plt.show()
        cv2.imwrite("data/mappiece/all_out.png", ground.map.map)


