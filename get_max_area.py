import os
import cv2
import numpy as np


def cvshow(img, title="image"):
    """显示图像并等待用户按键关闭窗口
    这个函数在调试过程中非常有用，可以在处理的各个阶段查看图像"""
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_centroid(contour):
    """计算轮廓的重心
    重心可以用来表示物体的中心位置，对于后续的物体跟踪或分析很有用"""
    M = cv2.moments(contour)
    if M['m00'] != 0:  # 防止除以零错误
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        return cX, cY
    return None


def calc_max_area(file_path):
    # 读取图像
    img = cv2.imread(file_path)
    if is_debug:
        cvshow(img)  # 显示原始图像，便于对比后续处理效果

    # 转换为灰度图，简化后续处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if is_debug:
        cvshow(gray)

    # 应用高斯模糊以减少噪声，提高角点检测的准确性
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if is_debug:
        cvshow(gray,"GaussianBlur")

    # 应用哈里斯角点检测，找出图像中的显著特征点
    gray_float = np.float32(gray)  # Harris角点检测需要float32类型的输入
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)

    # 结果图像的膨胀，使角点更加明显
    dst = cv2.dilate(dst, None)

    # 设定阈值，并标记角点，创建二值图像
    threshold = 0.01 * dst.max()
    black_image = np.zeros_like(gray)
    black_image[dst > threshold] = 255  # 角点位置设为白色
    if is_debug:
        cvshow(black_image, "black_image")

    # 执行开运算去除噪声，保留主要特征
    kernel = np.ones((4, 4), np.uint8)
    black_image = cv2.morphologyEx(black_image, cv2.MORPH_OPEN, kernel)
    if is_debug:
        cvshow(black_image, "kai")

    # 实现"浸染"效果，扩大特征区域
    radius = 3
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    dilated_image = cv2.dilate(black_image, kernel)
    if is_debug:
        cvshow(dilated_image, "qinran")

    # 将扩展区域的结果合并回原图像，形成更加连续的特征区域
    # black_image[dilated_image > 0] = 255
    # if is_debug:
    #     cvshow(black_image, "hebing")

    # 反转二值图像，准备进行轮廓检测
    binary = cv2.bitwise_not(dilated_image)
    if is_debug:
        cvshow(binary, "bitwise_not")

    # 查找轮廓，RETR_EXTERNAL只检测外部轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓，通常对应图像中最显著的特征
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if is_debug:
        # 在原图像上绘制所有轮廓，用于可视化
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
        cvshow(contour_img, "all lunkuo")

    # 在最大轮廓上绘制红色圆点标记重心
    if max_contour is not None:
        centroid = calculate_centroid(max_contour)
        if centroid:
            print("坐标：", centroid)
            cv2.circle(img, centroid, 5, (0, 0, 255), -1)  # 红色点
        cvshow(img, "max liantong")


# 设置调试模式，允许显示中间处理结果
is_debug = True

# 处理 "./images" 目录下的所有图像
files = os.listdir("./images")
for file in files:
    image_path = os.path.join("images", file)
    calc_max_area(image_path)