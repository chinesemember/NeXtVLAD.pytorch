import os
import numpy as np


def feature_pca_whiten(feat, center, eigenvals, eigenvecs):
    epsilon = 1e-4
    d = feat.shape[0]

    # subtract mean
    fcen = feat - center
    # principal components
    fpca = fcen.reshape((1, d)).dot(eigenvecs.T).squeeze(0)
    # whiten
    pcaw = fpca / np.sqrt(eigenvals + epsilon)

    return pcaw


def feature_pca(feat, center, eigenvals, eigenvecs):
    """
    Skip whitening, as done by Lin et al.
    :param feat:
    :param center:
    :param eigenvals:
    :param eigenvecs:
    :return:
    """
    d = feat.shape[0]

    # subtract mean
    fcen = feat - center
    # principal components
    fpca = fcen.reshape((1, d)).dot(eigenvecs.T).squeeze(0)

    return fpca
