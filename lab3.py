import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft
from scipy.stats import multivariate_normal
import random
from datetime import datetime

def kmeans_segm(image, K, L, seed = 42):
    """
    Implement a function that uses K-means to find cluster 'centers'
    and a 'segmentation' with an index per pixel indicating with 
    cluster it is associated to.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        seed - random seed
    Output:
        segmentation: an integer image with cluster indices
        centers: an array with K cluster mean colors
    """

    np.random.seed(seed)

    min_0 = np.min(image[:, :, 0])
    min_1 = np.min(image[:, :, 1])
    min_2 = np.min(image[:, :, 2])
    max_0 = np.max(image[:, :, 0])
    max_1 = np.max(image[:, :, 1])
    max_2 = np.max(image[:, :, 2])

    segmentation = np.zeros((image.shape[0], image.shape[1]))

    #stacking by rows: column = n % n_column  (row,column) = (n // n_column, n % n_column)
    colors_points = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    #colors_points_clusters = np.zeros(image.shape[0] * image.shape[1])

    centers = np.zeros((K,3))
    for i in range(len(centers)):
        centers[i] = [np.random.uniform(min_0, max_0), np.random.uniform(min_1, max_1),  np.random.uniform(min_2, max_2)]


    for i in range(L):
        #[number of pixels, K]
        distances = distance_matrix(colors_points, centers)
        colors_points_clusters = np.argmin(distances, axis=1)
        old_centers = np.copy(centers)
        for c in range(K):
            if (colors_points_clusters == c).any():
                centers[c] = np.mean(colors_points[colors_points_clusters == c], axis=0)
        print(i)
        if np.abs(np.sum(centers - old_centers)) < 0.5:
            break
    segmentation = colors_points_clusters.reshape(image.shape[0],image.shape[1])

    return segmentation, centers


def mixture_prob(image, K, L, mask):
    """
    Implement a function that creates a Gaussian mixture models using the pixels 
    in an image for which mask=1 and then returns an image with probabilities for
    every pixel in the original image.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        mask - an integer image where mask=1 indicates pixels used
    Output:
        prob: an image with probabilities per pixel
    """

    colors_points = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    mask_points = mask.reshape(image.shape[0] * image.shape[1])

    masked_color_points = colors_points[mask_points == 1]

    w = np.ones(K)
    mu = np.zeros((K,3))
    random.seed(datetime.now().timestamp())
    while True:
        s = int((100 * random.random()))
        #segmentation, mu = kmeans_segm(image, K, L, seed=100)
        segmentation, mu = kmeans_segm(np.copy(masked_color_points)[None, :, :], K, L, seed=s)
        for i in range(K):
            w[i] = len(np.where(segmentation == i)[0])
        #w /= (image.shape[0] * image.shape[1])
        w /= (masked_color_points.shape[0])
        if (w != 0).all():
            break

    # min_0 = np.min(masked_color_points[:, 0])
    # min_1 = np.min(masked_color_points[:, 1])
    # min_2 = np.min(masked_color_points[:, 2])
    # max_0 = np.max(masked_color_points[:, 0])
    # max_1 = np.max(masked_color_points[:, 1])
    # max_2 = np.max(masked_color_points[:, 2])
    #
    # for i in range(len(mu)):
    #     mu[i] = [np.random.uniform(min_0, max_0), np.random.uniform(min_1, max_1),  np.random.uniform(min_2, max_2)]
    # w = w/K

    sigma = 500*np.repeat(np.identity(3)[None, : , :], K, axis=0)

    N = len(masked_color_points)
    p_ik = np.zeros((N, K))

    print("w: ")
    print(w)
    for i in range(L):
        tot = 0
        for j in range(K):
            p_ik[:, j] = w[j] * multivariate_normal.pdf(masked_color_points, mean=mu[j], cov=sigma[j])
        #print(np.sum(p_ik, axis=0))
        for j in range(K):
            p_ik[:, j] /= np.sum(p_ik, axis=1)

        for j in range(K):
            mu[j] = (masked_color_points.T @ p_ik[:, j]) / np.sum(p_ik[:, j])
        w = np.mean(p_ik, axis=0)

        for j in range(K):
            x = (masked_color_points - mu[j].reshape(1, 3))[:, :, None]
            x_T = x.reshape(N, 1, 3)
            sigma[j] = np.sum((x @ x_T) * p_ik[:, j][:, None, None], axis=0) / np.sum(p_ik[:, j]) + np.identity(3) * 1e-6
            #sigma[j] *= np.identity(3)
        #
        # for j in range(K):
        #     if np.linalg.det(sigma[j]) <= 1e-10:
        #         sigma[j] = 50* np.identity(3)
        #         mu[j] = [np.random.uniform(min_0, max_0), np.random.uniform(min_1, max_1),  np.random.uniform(min_2, max_2)]



        #3xk / 1xk --> mu = 3xk
        #mu = ((masked_color_points.T @ p_ik) / np.sum(p_ik)).T
        #sigma
        # x = np.repeat(masked_color_points[None, :, :], K, axis=0) - mu.reshape(K, 1, 3)
        # x_T = x.reshape(K, 3, N)
        # p_ik_stack = np.repeat(p_ik[:, :, None], 3, axis=2).reshape(K, N, 3)
        # sigma = x_T @ (p_ik_stack * x) / np.sum(p_ik, axis=0)[:, None, None]

    prob = np.zeros((image.shape[0], image.shape[1]))
    for j in range(K):
        prob += w[j] * multivariate_normal.pdf(image, mean=mu[j], cov=sigma[j])
    return prob
