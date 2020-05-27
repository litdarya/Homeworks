from __future__ import print_function
from sys import argv
import os.path

import cv2
import numpy as np


def otsu(src_path, dst_path):
    img = cv2.imread(src_path)
    assert img is not None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bins = np.append(np.unique(gray.flatten()), 256)
    hist, bins = np.histogram(gray.flatten(), bins=bins, density=False)
    bins = bins[:-1]

    sum_height = np.sum(hist)
    sum_bins = np.sum(bins * hist)
    bins = bins[:-1]
    hist = hist[:-1]

    first_class_prob = np.cumsum(hist)
    second_class_prob = np.ones_like(first_class_prob)*sum_height - first_class_prob
    first_class_mean = np.true_divide(np.cumsum(bins*hist), first_class_prob)
    second_class_mean = np.true_divide(sum_bins - np.cumsum(bins*hist), second_class_prob)

    first_class_prob = np.true_divide(first_class_prob, sum_height)
    second_class_prob = np.true_divide(second_class_prob, sum_height)

    variance = np.power(first_class_mean - second_class_mean, 2)
    variance = first_class_prob*second_class_prob*variance

    best_t = bins[np.argmax(variance)]

    # print(best_t)

    gray[gray >= best_t] = 255
    gray[gray < best_t] = 0
    cv2.imwrite(dst_path, gray)


if __name__ == '__main__':
    assert len(argv) == 3
    assert os.path.exists(argv[1])
    otsu(*argv[1:])
