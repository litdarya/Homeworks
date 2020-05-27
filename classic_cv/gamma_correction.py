from sys import argv
import os.path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def gamma_correction(src_path, dst_path, a, b):
    src_img = cv2.imread(src_path)
    assert src_img is not None
    rescaled = src_img.astype(np.uint8) / 255.
    gamma = np.rint(255.0 * a * (np.power(rescaled, b))).astype(np.uint8)
    cv2.imwrite(dst_path, gamma)


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    gamma_correction(*argv[1:])
