from typing import Tuple

import numpy as np


def random_crop(image: np.ndarray, scale: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width, _ = image.shape
    new_height = round(height * scale)
    new_width = round(width * scale)
    offset_y = np.random.randint(0, height - new_height)
    offset_x = np.random.randint(0, width - new_width)
    image = image[offset_y:offset_y + new_height, offset_x:offset_x + new_width]
    return (image, (offset_y, offset_x))
