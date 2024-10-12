import os
import shutil
from collections import Counter

import rawpy
import cv2
import dlib
import numpy as np
from imutils import face_utils
import logging


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m',  # Magenta
    }

    RESET = '\033[0m'

    def format(self, record):
        log_fmt = f"{self.COLORS.get(record.levelname, self.RESET)}%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s{self.RESET}"
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


# 创建一个日志记录器
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

# 设置自定义的颜色格式化器
formatter = ColoredFormatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# 初始化dlib的人脸检测器和形状预测器模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')


def eye_aspect_ratio(eye):
    # 计算眼睛的长宽比
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def is_eyes_closed(eye):
    EAR_THRESHOLD = 0.2  # 根据实际情况调整此值
    ear = eye_aspect_ratio(eye)
    return ear < EAR_THRESHOLD


def get_image(file_path):
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        return cv2.imread(file_path)
    if file_path.lower().endswith('.cr3'):
        with rawpy.imread(file_path) as raw:
            return raw.postprocess()
    logger.debug(f'not supported file type: {file_path}')
    return None


def resize_image(image, max_width=1920, max_height=1080):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 如果图像的宽高都小于等于最大宽高，则不压缩
    if width <= max_width and height <= max_height:
        return image

    # 计算缩放比例
    scale = min(max_width / width, max_height / height)

    # 计算新的宽度和高度
    new_width = int(width * scale)
    new_height = int(height * scale)

    logger.info(f'resize image {width}x{height} to {new_width}x{new_height}, scale={scale}')

    # 调整图像大小
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def process_image(file_path):
    image = get_image(file_path)
    if image is None:
        logger.debug(f"Cannot read image: {file_path}")
        return -1
    image = resize_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    face_num = len(faces)
    if face_num == 0:
        logger.info(f'No face detected: {file_path}')
        return -2
    if face_num >= 3:
        logger.warning(f'More than 2 face detected, not support: {file_path}');
        return -3

    has_eye_close = has_any_eye_close(faces, gray)

    logger.info(f'Close eye {"" if has_eye_close else "not "}detected: {file_path}')
    if has_eye_close:
        logger.info(f'Eye closed, will mark {file_path} score as 1')
        mark_score_1(file_path)
    return 1 if has_eye_close else 0

def has_any_eye_close(faces, gray):
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[36:42]
        rightEye = shape[42:48]

        left_closed = is_eyes_closed(leftEye)
        right_closed = is_eyes_closed(rightEye)
        if left_closed or right_closed:
            return True
    return False


def mark_score_1(photo_path):
    # 使用 exiftool 修改 EXIF 信息
    xmp_file = os.path.splitext(photo_path)[0] + '.xmp'
    shutil.copy2('templates/mark1.xmp', xmp_file)

def scan_directory_and_score_images(directory):
    results = {}
    # 扫描指定目录及其子目录
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.cr3')):
                continue
            file_path = os.path.join(root, file)
            logger.debug(f"process {file_path} start")
            results[file_path] = process_image(file_path)
    return results

def main(directory):
    # 遍历目录中的所有JPG文件
    results = scan_directory_and_score_images(directory)
    # 映射结果代码到错误原因
    error_reasons = {
        0: "未检测到闭眼",
        1: "检测到闭眼",
        -1: "图像读取失败",
        -2: "没有检测到人脸",
        -3: "大于2张人脸"
    }

    # 统计每种结果的个数
    result_counts = Counter(results.values())

    # 计算总数
    total_count = len(results)

    # 打印每种结果的个数和占比
    logger.info("结果统计:")
    for result in sorted(result_counts.keys()):  # 排序结果代码
        count = result_counts[result]
        percentage = (count / total_count) * 100  # 计算占比
        reason = error_reasons.get(result, "未知原因")  # 获取对应的错误原因
        logger.info(f"结果 {result} ({reason}): 个数 = {count}, 占比 = {percentage:.2f}%")


if __name__ == "__main__":
    logger.info('start')
    directory = 'data/'
    main(directory)
