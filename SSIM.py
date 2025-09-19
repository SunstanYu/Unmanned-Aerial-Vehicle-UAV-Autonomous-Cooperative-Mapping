
import numpy as np
import cv2

def reorder_image(img, input_order='HWC', output_order='HWC'):
    ''' reorder_image '''
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if output_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong output_order {output_order}. Supported output_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img, input_order = img[..., None], 'CHW'
    if input_order == 'CHW' and output_order == 'HWC':
        img = img.transpose(1, 2, 0)
    elif input_order == 'HWC' and output_order == 'CHW':
        img = img.transpose(2, 0, 1)
    return img

def _convert_input_type_range(img):
    ''' convert input to [0, 1] '''
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img

def _convert_output_type_range(img, dst_type):
    ''' convert output to dst_type '''
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)

def bgr2ycbcr(img, y_only=False):
    ''' bgr space to ycbcr space '''
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def _ssim(img, img2):
    ''' ssim '''
    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **_kwargs):
    ''' calculate_ssim '''
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    if not isinstance(crop_border, (list, tuple)):
        crop_border = (crop_border, crop_border)
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    if crop_border[0] != 0:
        img = img[crop_border[0]:-crop_border[0], ...]
        img2 = img2[crop_border[0]:-crop_border[0], ...]
    if crop_border[1] != 0:
        img = img[:, crop_border[1]:-crop_border[1], ...]
        img2 = img2[:, crop_border[1]:-crop_border[1], ...]
    if test_y_channel:
        img = bgr2ycbcr(img.astype(np.float32) / 255., y_only=True)[..., None] * 255
        img2 = bgr2ycbcr(img2.astype(np.float32) / 255., y_only=True)[..., None] * 255
    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    ssims_result = np.array(ssims).mean()
    return ssims_result

if __name__ == '__main__':
    img1 = cv2.imread("data/random_1.jpg")  # 读入图片1
    img2 = cv2.imread("data/random_2.jpg")  # 读入图片2

    ssims_result = calculate_ssim(img1, img2, 4)

    print("PSNR_result = ", ssims_result)
