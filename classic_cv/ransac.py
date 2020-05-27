from __future__ import print_function
from sys import argv
import os.path, json
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2


def generate_line(img_size, line_params, N):
    w, h, = img_size
    a, b, c = line_params

    y_lim = abs(-(a * w + c) / b)

    if y_lim <= h:
        x_lim = w
    else:
        x_lim = abs(-(b * h + c) / a)

    x = np.arange(0, x_lim, x_lim / N)
    y = -(c + a * x) / b

    return x, y


def generate_data(img_size, line_params, n_points, sigma, inlier_ratio):
    normal_points = int(inlier_ratio*n_points)
    uniform_points = n_points - normal_points

    w, h, = img_size
    a, b, c = line_params

    x, y = generate_line(img_size, line_params, normal_points)

    x += np.random.normal(0, sigma, size=x.shape)
    y += np.random.normal(0, sigma, size=y.shape)

    idx_x = x < w
    idx_y = np.abs(y) < h
    idx = np.logical_and(idx_x, idx_y)
    x = x[idx][:normal_points]
    y = y[idx][:normal_points]

    if a/b > 0:
        h = -h

    x_line_center = (x[0] + x[-1]) / 2

    y_center = (-c - a * x_line_center) / b
    y_low = y_center - h / 2
    x_uniform = np.random.uniform(0, w, uniform_points)

    y_uniform = y_low + np.random.uniform(0, h, uniform_points)

    X = np.concatenate((x, x_uniform))
    Y = np.concatenate((y, y_uniform))

    return X, Y


def compute_ransac_thresh(alpha, sigma):
    return chi2.ppf(alpha, df=2)*sigma


def compute_ransac_iter_count(conv_prob, inlier_ratio):
    return np.log(1 - conv_prob)/np.log(1 - np.power(inlier_ratio, 2))


def mnk(X, y):
    A = np.hstack((X, np.ones_like(X)))
    return np.linalg.lstsq(A, y, rcond=None)[0]


def evaluate(X, y, model, t):
    Y = y.reshape((y.shape[0], 1))
    A = np.hstack((Y, X, np.ones_like(X)))
    model = np.insert(model, 0, -1.)

    distances = np.abs(np.sum(A * model, axis=1))
    distances /= np.sqrt(np.sum(np.power(model[:-1], 2)))

    return np.count_nonzero(distances <= t)


def compute_line_ransac(data, t, n):
    X, Y = data
    X = X.reshape((X.shape[0], 1))
    num_samples = X.shape[0]
    m = 2

    best_model = None
    best_score = 0

    for i in range(int(n)):
        sample = np.random.choice(num_samples, size=m, replace=False)
        params = mnk(X[sample], Y[sample])
        score = evaluate(X, Y, params, t)

        if score > best_score:
            best_model = params
            best_score = score

    return best_model


def main():
    print(argv)
    assert len(argv) == 2
    assert os.path.exists(argv[1])

    with open(argv[1]) as fin:
        params = json.load(fin)

    """
    params:
    line_params: (a,b,c) - line params (ax+by+c=0)
    img_size: (w, h) - size of the image
    n_points: count of points to be used

    sigma - Gaussian noise
    alpha - probability of point is an inlier

    inlier_ratio - ratio of inliers in the data
    conv_prob - probability of convergence
    """

    data = generate_data((params['w'], params['h']),
                         (params['a'], params['b'], params['c']),
                         params['n_points'], params['sigma'],
                         params['inlier_ratio'])

    t = compute_ransac_thresh(params['alpha'], params['sigma'])
    n = compute_ransac_iter_count(params['conv_prob'], params['inlier_ratio'])

    detected_line = compute_line_ransac(data, t, n)
    print(detected_line)
    print(f'y = {detected_line[0]}x + {detected_line[1]}')

    line = generate_line((params['w'], params['h']),
                         (-detected_line[0], 1, -detected_line[1]),
                         params['n_points']//2)
    perfect_line = generate_line((params['w'], params['h']),
                         (params['a'], params['b'], params['c']),
                         params['n_points'] // 2)
    plt.scatter(data[0], data[1], label='generated')
    plt.plot(line[0], line[1], color='red',
                label='estimated')
    plt.plot(perfect_line[0], perfect_line[1],
                color='green', label='perfect')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
