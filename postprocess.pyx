# cython: profile=True
from typing import List

import numpy as np
cimport numpy as np
import cv2
import cython


cdef inline np.ndarray[np.long_t, ndim=2] _neighbors(int y, int x):
    cdef:
        np.ndarray[np.long_t, ndim=2] a = np.empty((8, 2), np.long)
        int index = 0
        int i
        int j
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            a[index, 0] = y + i
            a[index, 1] = x + j
            index += 1
    return a


cdef inline int _is_valid_coor(int y, int x, int h, int w):
    return 0 <= y < h and 0 <= x < w


cdef inline (int, int) _find_root(dict union_find, (int, int) p):
    cdef:
        (int, int) parent = union_find.get(p, p)
        (int, int) root
    if parent[0] == p[0] and parent[1] == p[1]:
        union_find[p] = p
        return p
    root = _find_root(union_find, parent)
    union_find[p] = root
    return root


cdef inline void _link(dict union_find, (int, int) p1, (int, int) p2):
    cdef:
        (int, int) root1 = _find_root(union_find, p1)
        (int, int) root2 = _find_root(union_find, p2)
    if root1[0] != root2[0] or root1[1] != root2[1]:
        union_find[root1] = root2


@cython.boundscheck(False)
cdef np.ndarray[np.int32_t, ndim=2] _link_pixels(np.ndarray[np.long_t, ndim=2] pixel_mask, np.ndarray[np.uint8_t, ndim=3] link_mask):
    """Link positive pixels.

    Args:
        pixel_mask: uint8 array of shape [H, W]
        link_mask: uint8 array of shape [8, H, W]
    Returns:
        link_map: int array of shape [H, W]. If the pixel (x, y)
                  is linked to i-th group, link_map[y, x] = i.
    """
    cdef:
        dict union_find = {}
        int mask_height = pixel_mask.shape[0]
        int mask_width = pixel_mask.shape[1]
        np.ndarray[np.long_t, ndim=1] ys, xs  # points of text area
        np.ndarray[np.long_t, ndim=1] point
        int index, x, y, x_, y_, i, root_index
        int neighbor_index, dy, dx
        np.ndarray[np.int32_t, ndim=2] res
        dict roots
    # for point in points:
    ys, xs = np.where(pixel_mask)
    for index in range(ys.shape[0]):
        y = ys[index]
        x = xs[index]
        neighbor_index = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                y_ = y + dy
                x_ = x + dx
                if _is_valid_coor(y_, x_, mask_height, mask_width) and link_mask[y_, x_, neighbor_index] == 1:
                    _link(union_find, (y, x), (y_, x_))
                neighbor_index += 1

    res = np.zeros((mask_height, mask_width), np.int32)
    roots = {}
    for index in range(ys.shape[0]):
        y = ys[index]
        x = xs[index]
        root = _find_root(union_find, (y, x))
        root_index = roots.get(root, -1)
        if root_index == -1:
            root_index = len(roots) + 1
            roots[root] = root_index
        res[y, x] = root_index
    return res


cpdef np.ndarray[np.uint8_t, ndim=3] mask_to_instance_map(np.ndarray[np.float32_t, ndim=3] pixel_pred, np.ndarray[np.float32_t, ndim=3] link_mask):
    cdef:
        int mask_height = pixel_pred.shape[0]
        int mask_width = pixel_pred.shape[1]
        np.ndarray[np.long_t, ndim=2] pixel_mask = np.argmax(pixel_pred, axis=2)
        np.ndarray[np.uint8_t, ndim=3] link_neighbors = np.zeros((mask_height, mask_width, 8), dtype=np.uint8)
        int i
        np.ndarray[np.long_t, ndim=2] neighbor
    for i in range(8):
        neighbor = np.argmax(link_mask[:, :, 2*i:2*(i+1)], axis=2)
        link_neighbors[:, :, i] = neighbor.astype(np.uint8)
    link_neighbors = link_neighbors * np.expand_dims(pixel_mask, 2).astype(np.uint8)
    return _link_pixels(pixel_mask, link_neighbors)


def instance_map_to_bboxes(instance_map: np.ndarray) -> List[List[np.ndarray]]:
    map_height = instance_map.shape[0]
    map_width = instance_map.shape[1]
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
