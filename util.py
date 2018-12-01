from typing import Tuple
import math

import numpy as np
import cv2


def rotate(image: np.ndarray, angle: int) -> np.ndarray:
    assert angle in [0, 90, 180, 270]
    if angle == 90:
        image = cv2.transpose(image)
        image = cv2.flip(image, 1)
    elif angle == 180:
        image = cv2.flip(image, -1)
    elif angle == 270:
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)
    return image


def random_crop(image: np.ndarray, scale: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width, _ = image.shape
    new_height = round(height * scale)
    new_width = round(width * scale)
    offset_y = np.random.randint(0, height - new_height)
    offset_x = np.random.randint(0, width - new_width)
    image = image[offset_y:offset_y + new_height, offset_x:offset_x + new_width]
    return (image, (offset_y, offset_x))


def transform_aspect_ratio(image: np.ndarray, aspect_ratio: float) -> np.ndarray:
    original_height, original_width, _ = image.shape
    area = original_height * original_width
    height = round(math.sqrt(area / aspect_ratio))
    width = round(area / height)
    return cv2.resize(image, (round(width), round(height)))
