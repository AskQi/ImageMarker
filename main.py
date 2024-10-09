import os
import shutil

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
logger = logging.getLogger('my_logger')
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


def process_image(file_path):
    # 读取图像
    with rawpy.imread(file_path) as raw:
        image = raw.postprocess()
    scale_percent = 20  # 缩小到50%
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    face_num = len(faces)
    if face_num == 0:
        logger.info(f'No face detected: {file_path}')
        return
    if face_num >= 3:
        logger.warning(f'More than 2 face detected, not support: {file_path}');
        return

    has_eye_close = False
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[36:42]
        rightEye = shape[42:48]

        left_closed = is_eyes_closed(leftEye)
        right_closed = is_eyes_closed(rightEye)
        if left_closed or right_closed:
            has_eye_close = True
            break

    logger.info(f'Close eye {"" if has_eye_close else "not "}detected: {file_path}')
    if has_eye_close:
        logger.info(f'Eye closed, will mark {file_path} score as 1')
        mark_score_1(file_path)


def mark_score_1(photo_path):
    # 使用 exiftool 修改 EXIF 信息
    xmp_file = os.path.splitext(photo_path)[0] + '.xmp'
    shutil.copy2('templates/mark1.xmp', xmp_file)


def main(directory):
    # 遍历目录中的所有JPG文件
    for filename in os.listdir(directory):
        if filename.lower().endswith('.cr3'):
            file_path = os.path.join(directory, filename)
            logger.debug(f"process {file_path} start")
            process_image(file_path)


if __name__ == "__main__":
    logger.info('start')
    directory = 'data/'
    main(directory)
