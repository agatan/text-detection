from icecream import ic
import torch
import numpy as np


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
            if is_valid_coor(y_, x_, mask_height, mask_width) and link_mask[i, y_, x_] == 1:
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


def mask_to_instance_map(pixel_mask: torch.Tensor, link_mask: torch.Tensor) -> np.ndarray:
    batch_size, _, mask_height, mask_width = pixel_mask.size()

    _, pixel_mask = torch.max(pixel_mask, dim=1)
    link_neighbors = torch.zeros(batch_size, 8, mask_height, mask_width, dtype=torch.uint8)
    for i in range(8):
        _, neighbor = torch.max(link_mask[:, 2*i:2*(i+1)], dim=1)
        link_neighbors[:, i] = neighbor
    link_neighbors = link_neighbors * pixel_mask.unsqueeze(1).byte()

    pixel_mask = pixel_mask.cpu().numpy()
    link_neighbors = link_neighbors.cpu().numpy()
    instance_maps = []
    for i in range(batch_size):
        instance_maps.append(_link_pixels(pixel_mask[i], link_neighbors[i]))
    return np.stack(instance_maps, axis=0)
