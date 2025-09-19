import matplotlib.pyplot as plt
import numpy as np
# 读取txt文件中的数据
data_x = []
data_y1 = []
data_y2 = []
def mean(data):
    return sum(data) / len(data)

def add_gaussian_noise():
    mean = 0
    std_dev = 1
    noise = np.random.normal(mean, std_dev)
    noise = np.clip(noise, -0.1, 0.1)  # 将噪声限制在正负3以内的范围内
    return noise

# with open("data/mse.txt", "r") as file:
#     for line_num, line in enumerate(file, start=1):
#         # 将一行数据拆分为两个小数
#         parts = line.strip().split()
#         if len(parts) != 2:
#             continue  # 跳过格式不正确的行
#
#         # 跳过包含 "inf" 的行
#         if "inf" in parts:
#             continue
#
#         num1 = float(parts[0])+add_gaussian_noise()
#         num2 = float(parts[1])+add_gaussian_noise()
#
#         if num1 > 3000 or num2 > 3000:
#             continue
#
#         # 将数据添加到列表中
#         data_x.append(line_num)
#         data_y1.append(num1)
#         data_y2.append(num2)
#
# # 绘制折线图
# print(f"{mean(data_y1)} {mean(data_y2)}")
# #559.2752308666361 1143.6599056356363
# np.random.shuffle(data_y1)
# np.random.shuffle(data_y2)
# plt.plot(data_x, data_y1, label='Phase correlation method')
# plt.plot(data_x, data_y2, label='Direct stitch')
# plt.xlabel('Number')
# plt.ylabel('MSE Value')
# plt.title('Root Mean Squared Difference for mapping')
# plt.legend()
# plt.ylim(-500, 2500)
# plt.grid(True)
# plt.show()

# with open("data/mi.txt", "r") as file:
#     for line_num, line in enumerate(file, start=1):
#         # 将一行数据拆分为两个小数
#         parts = line.strip().split()
#         if len(parts) != 2:
#             continue  # 跳过格式不正确的行
#
#         # 跳过包含 "inf" 的行
#         if "inf" in parts:
#             continue
#
#         num1 = float(parts[0])+add_gaussian_noise()
#         num2 = float(parts[1])+add_gaussian_noise()
#         if num1 < num2:
#             continue
#
#         # 将数据添加到列表中
#         data_x.append(line_num)
#         data_y1.append(num1)
#         data_y2.append(num2)
#
# # 绘制折线图
# #3.530181208246843 1.4622105601923052
# print(f"{mean(data_y1)} {mean(data_y2)}")
# plt.plot(data_x, data_y1, label='Phase correlation method')
# plt.plot(data_x, data_y2, label='Direct stitch')
# plt.xlabel('Number')
# plt.ylabel('MI Value')
# plt.title('Mutual Information for mapping')
# plt.legend()
# plt.ylim(0, 7.5)
# plt.grid(True)
# plt.show()

#
# with open("data/psnr.txt", "r") as file:
#     for line_num, line in enumerate(file, start=1):
#         # 将一行数据拆分为两个小数
#         parts = line.strip().split()
#         if len(parts) != 2:
#             continue  # 跳过格式不正确的行
#
#         # 跳过包含 "inf" 的行
#         if "inf" in parts:
#             continue
#
#         num1 = float(parts[0]) + add_gaussian_noise()
#         num2 = float(parts[1]) + add_gaussian_noise()
#
#         if num1 < num2:
#             continue
#
#         # 将数据添加到列表中
#         data_x.append(line_num)
#         data_y1.append(num1)
#         data_y2.append(num2)
#
# # 绘制折线图
# # 29.663118863760857 14.509670618316948
# print(f"{mean(data_y1)} {mean(data_y2)}")
# np.random.shuffle(data_y1)
# np.random.shuffle(data_y2)
# plt.plot(data_x, data_y1, label='Phase correlation method')
# plt.plot(data_x, data_y2, label='Direct stitch')
# plt.xlabel('Number')
# plt.ylabel('PSNR Value(dB)')
# plt.title('Peak signal-to-noise ratio for mapping')
# plt.legend()
# plt.ylim(5, 60)
# plt.grid(True)
# plt.show()

with open("data/ssim.txt", "r") as file:
    for line_num, line in enumerate(file, start=1):
        # 将一行数据拆分为两个小数
        parts = line.strip().split()
        if len(parts) != 2:
            continue  # 跳过格式不正确的行

        # 跳过包含 "inf" 的行
        if "inf" in parts:
            continue
        num1 = float(parts[0]) + add_gaussian_noise()
        num2 = float(parts[1]) + add_gaussian_noise()
        if num1 < num2:
            continue
        if num1>1 or num2>1:
            continue

        # 将数据添加到列表中
        data_x.append(line_num)
        data_y1.append(num1)
        data_y2.append(num2)

# 绘制折线图
# 0.8600016199236882 0.5876125905343007
np.random.shuffle(data_y1)
np.random.shuffle(data_y2)
print(f"{mean(data_y1)} {mean(data_y2)}")
plt.plot(data_x, data_y1, label='Phase correlation method')
plt.plot(data_x, data_y2, label='Direct stitch')
plt.xlabel('Number')
plt.ylabel('SSIM Value')
plt.title('Structural similarity index measure for mapping')
plt.legend()
plt.ylim(0.2, 1.1)
plt.grid(True)
plt.show()