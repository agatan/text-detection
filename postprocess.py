from typing import List, Tuple

import torch
import numpy as np
import cv2


def _link_pixels(pixel_mask: np.ndarray, link_mask: np.ndarray) -> np.ndarray:
    """Link positive pixels.
    Args:
        pixel_mask: uint8 array of shape [H, W]
        link_mask: uint8 array of shape [8, H, W]
    Returns:
        link_map: int array of shape [H, W]. If the pixel (x, y)
                  is linked to i-th group, link_map[y, x] = i.
    """
    union_find = {}

    def find_root(p):
        if union_find.get(p, p) == p:
            union_find[p] = p
            return p
        root = find_root(union_find[p])
        union_find[p] = root
        return root

    def link(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)
        if root1 != root2:
            union_find[root1] = root2

    def neighbors(y, x):
        return [
            (y - 1, x - 1),
            (y - 1, x),
            (y - 1, x + 1),
            (y, x - 1),
            (y, x + 1),
            (y + 1, x - 1),
            (y + 1, x),
            (y + 1, x + 1),
        ]

    def is_valid_coor(y, x, h, w):
        return 0 <= y < h and 0 <= x < w

    mask_height, mask_width = pixel_mask.shape
    points = list(zip(*np.where(pixel_mask)))  # points of text area
    for point in points:
        y, x = point
        for i, (y_, x_) in enumerate(neighbors(y, x)):
            if is_valid_coor(y_, x_, mask_height, mask_width) and link_mask[y_, x_, i] == 1:
                link(point, (y_, x_))

    res = np.zeros((mask_height, mask_width), np.int32)
    roots = {}
    for point in points:
        y, x = point
        root = find_root(point)
        root_index = roots.get(root, None)
        if root_index is None:
            root_index = len(roots) + 1
            roots[root] = root_index
        res[y, x] = root_index
    return res


def mask_to_instance_map(pixel_mask: np.ndarray, link_mask: np.ndarray, mask_threshold: float = 0.5, link_threshold: float = 0.5) -> np.ndarray:
    mask_height, mask_width, _ = pixel_mask.shape

    pixel_mask = np.argmax(pixel_mask, axis=2)
    max_class = np.max(pixel_mask)
    link_neighbors = np.zeros((mask_height, mask_width, 8), dtype=np.uint8)
    for i in range(8):
        neighbor = link_mask[:, :, 2 * i + 1] > link_threshold
        link_neighbors[:, :, i] = neighbor
    instance_map = np.zeros(pixel_mask.shape[:2], dtype=np.int32)
    class_map = np.zeros(pixel_mask.shape[:2], dtype=np.int32)
    sum_boxes = 0
    for cls in range(1, max_class + 1):
        class_mask = pixel_mask == cls
        class_link_neighbors = link_neighbors * np.expand_dims(class_mask, 2).astype(np.uint8)
        class_instances = _link_pixels(class_mask, class_link_neighbors)
        num_boxes = np.max(class_instances)
        class_map[class_instances != 0] = cls
        class_instances[class_instances != 0] += sum_boxes
        sum_boxes += num_boxes
        instance_map += class_instances
    return instance_map, class_map


def instance_map_to_bboxes(instance_map: np.ndarray) -> List[List[np.ndarray]]:
    map_height, map_width = instance_map.shape
    bounding_boxes = []
    num_bboxes = np.max(instance_map)
    bboxes = []
    for n in range(1, num_bboxes + 1):
        points = np.array(list(zip(*np.where(instance_map == n))))
        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect)
        bbox[:, 0] = np.clip(bbox[:, 0], 0, map_height)
        bbox[:, 1] = np.clip(bbox[:, 1], 0, map_width)
        bboxes.append(bbox)
    return bboxes
