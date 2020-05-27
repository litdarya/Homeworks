from __future__ import print_function
from sys import argv
import os.path

import cv2
import numpy as np


def two_dim_idx(idx, matrix_shape):
    row = lambda x: x // matrix_shape
    column = lambda x: x % matrix_shape

    rows_idx = row(idx)
    column_idx = column(idx)

    return rows_idx, column_idx


def min_max_dx(ratio, gray):
    idx = np.argpartition(gray, ratio + 1)
    max_col = gray[idx[ratio]]
    return idx[:ratio], abs(max_col)


def contrast_channel(img_part, white_perc, black_perc):
    white_ratio = int(img_part.shape[0] * img_part.shape[1] * white_perc)
    black_ratio = int(img_part.shape[0] * img_part.shape[1] * black_perc)

    white_idx, max_white = min_max_dx(ratio=white_ratio,
                                      gray=-1 * img_part.flatten())

    black_idx, min_black = min_max_dx(ratio=black_ratio,
                                      gray=img_part.flatten())

    linear = lambda x: 255. * (x - min_black) / (max_white - min_black)
    img_part = np.rint(linear(img_part)).astype(np.uint8)

    white_rows, white_cols = two_dim_idx(white_idx, img_part.shape[1])
    black_rows, black_cols = two_dim_idx(black_idx, img_part.shape[1])

    img_part[white_rows, white_cols] = 255
    img_part[black_rows, black_cols] = 0

    return img_part


def autocontrast(src_path, dst_path, white_perc, black_perc):
    img = cv2.imread(src_path)
    assert img is not None

    if len(img.shape) == 2:
        res = contrast_channel(img, white_perc, black_perc)
    else:
        b = contrast_channel(img[:, :, 0], white_perc, black_perc)
        g = contrast_channel(img[:, :, 1], white_perc, black_perc)
        r = contrast_channel(img[:, :, 2], white_perc, black_perc)
        res = np.stack((b, g, r), axis=-1)

    cv2.imwrite(dst_path, res)


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    assert 0 <= argv[3] < 1
    assert 0 <= argv[4] < 1

    autocontrast(*argv[1:])
