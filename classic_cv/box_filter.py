from __future__ import print_function
from sys import argv
import os.path

import cv2
import numpy as np


def calc_cum_sum_2d(a):
    if len(a.shape) == 3:
        # color mode
        a = np.pad(a, [(1, 0), (1, 0), (0, 0)], mode='constant')
    else:
        # grayscale mode
        a = np.pad(a, [(1, 0), (1, 0)], mode='constant')
    return np.cumsum(np.cumsum(a, axis=0), axis=1)


def get_sum_range(pref_sum, x1, x2, y1, y2):
    """
    sum in [x1; x2)x[y1; y2)
    """
    return (pref_sum[x2, y2]
            - pref_sum[x1, y2]
            - pref_sum[x2, y1]
            + pref_sum[x1, y1])


def get_area_size(i1, i2, j1, j2):
    return (i2 - i1) * (j2 - j1)


def box_flter(src_path, dst_path, w, h):
    src_img = cv2.imread(src_path).astype(np.int)
    assert src_img is not None

    pref_sum = calc_cum_sum_2d(src_img)

    n = len(src_img)
    m = len(src_img[0])
    if len(src_img.shape) > 2:
        mode = 'color'
    else:
        mode = 'grayscale'

    row_indices, col_indices = np.indices(src_img.shape[:2])

    width_padding = w // 2
    height_padding = h // 2

    x1 = np.maximum(0, row_indices - height_padding)
    x2 = np.minimum(n, row_indices + height_padding + 1)
    y1 = np.maximum(0, col_indices - width_padding)
    y2 = np.minimum(m, col_indices + width_padding + 1)

    area_size = get_area_size(x1, x2, y1, y2)
    if mode == 'color':
        area_size = area_size[:, :, np.newaxis]

    new_box = get_sum_range(pref_sum, x1, x2, y1, y2) / area_size
    new_box = new_box.astype(np.uint8)
    cv2.imwrite(dst_path, new_box)


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = int(argv[3])
    argv[4] = int(argv[4])
    assert argv[3] > 0
    assert argv[4] > 0

    box_flter(*argv[1:])
