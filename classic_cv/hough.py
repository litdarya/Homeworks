from __future__ import print_function
from sys import argv
import cv2
import numpy as np
from math import hypot
from matplotlib import pyplot as plt


def gradient_img(img):
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)

    return magnitude


def hough_transform(img, theta, rho):
    thetas = np.arange(0, np.pi, theta)
    max_distance = hypot(img.shape[0], img.shape[1])
    rhos = np.arange(-max_distance, max_distance, rho)

    row_indices, col_indices = np.indices(img.shape)
    row_indices_3d = np.repeat(row_indices[:, :, np.newaxis], len(thetas), axis=2)
    col_indices_3d = np.repeat(col_indices[:, :, np.newaxis], len(thetas), axis=2)

    R = col_indices_3d * np.sin(thetas) + row_indices_3d * np.cos(thetas)
    R_idx = ((R + max_distance) / rho).astype(np.int)

    ht_map = np.zeros((rhos.shape[0], thetas.shape[0]))
    theta_idx = np.tile(np.arange(len(thetas)), (img.shape[0], img.shape[1], 1))
    magnitude_3d = np.repeat(img[:, :, np.newaxis], len(thetas), axis=2)
    # magnitude_3d[magnitude_3d > 0] = 1
    np.add.at(ht_map, (R_idx.flatten(), theta_idx.flatten()), magnitude_3d.flatten())

    ht_map = np.rint(ht_map / np.max(ht_map.flatten()) * 255.).astype(np.int)

    return ht_map, thetas, rhos


def add_point(matrix, x, y, size=10, color=255):
    for i in np.arange(x - size, x + size):
        for j in np.arange(y - size, y + size):
            if 0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1]:
                # print(matrix.shape, i, j, color)
                matrix[i, j] = color
    return matrix


def add_line(matrix, x, y, color=(255, 0, 0)):
    for i, j in zip(x, y):
        matrix = add_point(matrix, i, j, color=color, size=2)
    return matrix


def get_n_the_most_bright(ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta):
    values = np.argsort(-1 * ht_map.flatten().copy())
    theta_idx_all = values % ht_map.shape[1]
    rho_idx_all = (values // ht_map.shape[1]).astype(np.int)

    theta_idx = np.array([], dtype=np.int)
    rho_idx = np.array([], dtype=np.int)

    i = 0
    while len(theta_idx) < n_lines and i < len(values):
        assert len(theta_idx) == len(rho_idx)
        is_suitable_theta = np.all(np.abs(thetas[theta_idx] - thetas[theta_idx_all[i]]) % (np.pi) > min_delta_theta)
        is_suitable_rho = np.all(np.abs(rhos[rho_idx] - rhos[rho_idx_all[i]]) > min_delta_rho)

        if is_suitable_rho or is_suitable_theta:
            theta_idx = np.append(theta_idx, theta_idx_all[i])
            rho_idx = np.append(rho_idx, rho_idx_all[i])

        i += 1

    return theta_idx, rho_idx


def get_lines(src_image, ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta):
    theta_idx, rho_idx = get_n_the_most_bright(ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta)

    if len(theta_idx) < n_lines:
        print()
        print(f"!!!!!!!! found only {len(theta_idx)} lines, expected {n_lines}")
        print()

    best_thetas = thetas[theta_idx]
    best_rhos = rhos[rho_idx]

    image_with_lines = src_image.copy()
    image_with_lines = np.repeat(image_with_lines[:, :, np.newaxis], 3, axis=2)

    params = []
    for r, theta in zip(best_rhos, best_thetas):
        X = np.arange(0, image_with_lines.shape[0])
        Y = np.arange(0, image_with_lines.shape[1])
        if np.abs(np.sin(theta)) > np.abs(np.cos(theta)):
            a = -np.cos(theta) / np.sin(theta)
            b = r / np.sin(theta)
            y = a * X + b
            print(f'найдена прямая y = {a}*x + {b}')
            params.append((a, b))
            # plt.plot(X, y)
            image_with_lines = add_line(image_with_lines, X, y.astype(np.int))
        else:
            a = r / np.cos(theta)
            b = np.sin(theta) / np.cos(theta)
            x = a - Y * b

            if abs(b) < 1e-2:
                print('ИЗ ЗАДАНИЯ НЕ ЯСНО, КАК ВЫПИСАТЬ Х=А')
                print(f'найдена прямая x = {a}')
                params.append((np.inf, np.inf))
            else:
                print(f'найдена прямая y = {-1./b}*x + {a/b}')
                params.append((-1./b, a/b))
            # plt.plot(x, Y)
            image_with_lines = add_line(image_with_lines, x.astype(np.int), Y)
    cv2.imwrite('image_with_lines.png', image_with_lines)
    # plt.show()
    return params


if __name__ == '__main__':
    assert len(argv) == 9
    src_path, dst_ht_path, dst_lines_path, theta, rho,\
        n_lines, min_delta_rho, min_delta_theta = argv[1:]

    theta = float(theta)
    rho = float(rho)
    n_lines = int(n_lines)
    min_delta_rho = float(min_delta_rho)
    min_delta_theta = float(min_delta_theta)

    assert theta > 0.0
    assert rho > 0.0
    assert n_lines > 0
    assert min_delta_rho > 0.0
    assert min_delta_theta > 0.0

    image = cv2.imread(src_path, 0)
    assert image is not None

    image = image.astype(float)
    gradient = gradient_img(image)

    ht_map, thetas, rhos = hough_transform(gradient, theta, rho)

    cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(image, ht_map, n_lines, thetas, rhos,
                      min_delta_rho, min_delta_theta)

    with open(dst_lines_path, 'w') as fout:
        for line in lines:
            fout.write('%0.3f, %0.3f\n' % line)
