"""
Re-tooled version of the script found on VideoToTextDNN:
https://github.com/OSUPCVLab/VideoToTextDNN/blob/master/data/py3_process_features.py

Perform batched feature extract using Cadene pretrainedmodels
"""
import torch
import argparse
import time
import os
import numpy as np
import logging

from util import TransformImage, create_batches, process_batches, init_model

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

available_features = ['nasnetalarge', 'resnet152', 'pnasnet5large', 'densenet121', 'senet154', 'polynet', 'vgg16']

args = None


def extract_features(args):
    root_frames_dir = args.frames_dir
    root_feats_dir = args.feats_dir
    work = args.work
    autofill = int(args.autofill)
    ftype = args.type
    gpu_list = args.gpu_list

    class_dirs = os.listdir(root_frames_dir)

    # skip a level for UCF101 dataset
    for class_dir in class_dirs:
        class_frames_dir = os.path.join(root_frames_dir, class_dir)

        frames_dirs = os.listdir(class_frames_dir)

        class_feats_dir = os.path.join(root_feats_dir, class_dir)
        if not os.path.isdir(class_feats_dir):
            os.makedirs(class_feats_dir)

        # else:
        #     if autofill:
        #         logger.info('AUTOFILL ON: Attempting to autofill missing features.')
        #         frames_dirs = validate_feats.go(featsd=root_feats_dir, framesd=root_frames_dir)

        # Difficulty of each job is measured by # of frames to process in each chunk.
        # Can't be randomized since autofill list woudld be no longer valid.
        # np.random.shuffle(frames_dirs)
        work = len(frames_dirs) if not work else work

        tf_img, model = init_model(args.gpu_list, args.type)

        work_done = 0
        while work_done != work:
            frames_dirs_avail = diff_feats(class_frames_dir, class_feats_dir)
            if len(frames_dirs_avail) == 0:
                break

            frames_dir = frames_dirs_avail.pop()
            feat_filename = frames_dir.split('/')[-1] + '.npy'
            video_feats_path = os.path.join(class_feats_dir, feat_filename)

            if os.path.exists(video_feats_path):
                logger.info('Features already extracted:\t{}'.format(video_feats_path))
                continue

            try:
                frames_to_do = [os.path.join(args.frames_dir, class_dir, frames_dir, p) for p in
                                os.listdir(os.path.join(args.frames_dir, class_dir, frames_dir))]
            except Exception as e:
                logger.exception(e)
                continue

            # Must sort so frames follow numerical order. os.listdir does not guarantee order.
            frames_to_do.sort()

            if len(frames_to_do) == 0:
                logger.warning("Frame folder has no frames! Skipping...")
                continue

            # Save a flag copy
            with open(video_feats_path, 'wb') as pf:
                np.save(pf, [])

            try:
                batches = create_batches(frames_to_do, tf_img, logger=logger, batch_size=args.batch_size)
            except OSError as e:
                logger.exception(e)
                logger.warning("Corrupt image file. Skipping...")
                os.remove(video_feats_path)
                continue

            logger.debug("Start video {}".format(work_done))

            feats = process_batches(batches, ftype, gpu_list, model, logger=logger)

            with open(video_feats_path, 'wb') as pf:
                np.save(pf, feats)
                logger.info('Saved complete features to {}.'.format(video_feats_path))
            work_done += 1


def diff_feats(frames_dir, feats_dir):
    feats = ['.'.join(i.split('.')[:-1]) for i in os.listdir(feats_dir)]
    feats = set(feats)
    frames = set([fr for fr in os.listdir(frames_dir) if len(os.listdir(os.path.join(frames_dir, fr)))])
    needed_feats = frames - feats
    return needed_feats


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('frames_dir',help = 'Directory where there are frame directories.')
    arg_parser.add_argument('feats_dir',help = 'Root directory of dataset\'s processed videos.')
    arg_parser.add_argument('-w', '--work', help = 'Number of features to process. Defaults to all.', default=0, type=int)
    arg_parser.add_argument('-gl', '--gpu_list', required=True, nargs='+', type=int, help="Space delimited list of GPU indices to use. Example for 4 GPUs: -gl 0 1 2 3")
    arg_parser.add_argument('-bs', '--batch_size', type=int, help="Batch size to use during feature extraction. Larger batch size = more VRAM usage", default=8)
    arg_parser.add_argument('--type', required=True, help = 'ConvNet to use for processing features.', choices=available_features)
    arg_parser.add_argument('--autofill', action='store_true', default=False, help="Perform diff between frames_dir and feats_dir and fill them in.")

    args = arg_parser.parse_args()

    start_time = time.time()

    logger.info("Found {} GPUs, using {}.".format(torch.cuda.device_count(), len(args.gpu_list)))

    extract_features(args)

    logger.info("Job took %s mins" % ((time.time() - start_time)/60))
