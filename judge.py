import cv2
import numpy as np


def MSE_calculate(ref_image, stitched_image):
    # 读取参考图像和拼接图像
    # ref_image = cv2.imread(ref_image_path)
    # stitched_image = cv2.imread(stitched_image_path)

    # 确保图像具有相同的维度
    assert ref_image.shape == stitched_image.shape, "Images must have the same dimensions"

    # 计算均方误差（MSE）
    mse = np.mean(np.square(np.subtract(ref_image.astype(np.float64), stitched_image.astype(np.float64))))

    return mse
    # # 根据MSE计算峰值信噪比（PSNR）
    # psnr = 20 * np.log10(255 / np.sqrt(mse))
    #
    # # 计算结构相似性指数（SSIM）
    # window_size = (11, 11)  # 高斯窗口大小
    # c1 = (2 * np.var(ref_image) + np.var(stitched_image)) * window_size[0] ** 2
    # c2 = 2 * np.var(ref_image) * np.var(stitched_image) * window_size[0] ** 2
    # mu_ref = np.mean(ref_image)
    # mu_stitched = np.mean(stitched_image)
    # sigma_ref = np.std(ref_image)
    # sigma_stitched = np.std(stitched_image)
    #
    # ssim_numerator = (2 * sigma_ref * sigma_stitched + c1)
    # ssim_denominator = (sigma_ref ** 2 + sigma_stitched ** 2 + c1)
    # ssim = ssim_numerator / ssim_denominator
    #
    # # 显示评估结果
    # print('Mean Squared Error (MSE): {:.4f}'.format(mse))
    # print('Peak Signal to Noise Ratio (PSNR): {:.2f} dB'.format(psnr))
    # print('Structural Similarity Index (SSIM): {:.4f}'.format(ssim))


# 调用函数
# evaluate_stitching_quality("ref_image.jpg", "stitched_image.jpg")