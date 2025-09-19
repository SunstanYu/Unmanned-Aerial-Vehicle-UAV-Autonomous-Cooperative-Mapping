import cv2
import tkinter as tk


def show_img(result_img):
    if len(result_img.shape) == 2:
        h, w = result_img.shape
    else:
        h, w, _ = result_img.shape
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()  # 获取屏幕的宽度
    screen_height = root.winfo_screenheight()  # 获取屏幕的高度

    scale = min(screen_width / w, screen_height / h)  # 计算缩放比例
    img_resized = cv2.resize(result_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x = int((screen_width - img_resized.shape[1]) / 2)
    y = int((screen_height - img_resized.shape[0]) / 2)
    # 显示结果图像
    cv2.imshow('Image Matching Result', img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()