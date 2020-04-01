import os
import torch
import numpy as np
from PIL import Image


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


def create_batches(frames_to_do, tf_img_fn, logger=None, batch_size=32):
    n = len(frames_to_do)
    if n < batch_size:
        if logger: logger.warning("Sample size less than batch size: Cutting batch size.")
        batch_size = n

    if logger: logger.info("Generating {} batches...".format(n // batch_size))
    batches = []
    frames_to_do = np.array(frames_to_do)

    for idx in range(0, n, batch_size):
        frames_idx = list(range(idx, min(idx+batch_size, n)))
        batch_frames = frames_to_do[frames_idx]

        batch_tensor = torch.zeros((len(batch_frames),) + tuple(tf_img_fn.input_size))
        for i, frame_ in enumerate(batch_frames):
            input_frame = Image.fromarray(frame_).convert('RGB')
            input_tensor = tf_img_fn(input_frame)  # 3x400x225 -> 3x299x299 size may differ
            # input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            batch_tensor[i] = input_tensor

        batch_ag = torch.autograd.Variable(batch_tensor, requires_grad=False)
        batches.append(batch_ag)
    return batches
