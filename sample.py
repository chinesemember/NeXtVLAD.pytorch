import argparse
import os
import torch
import numpy as np
import logging
import ffmpeg

from models.video_classifiers import NeXtVLADModel
from util import feature_pca, create_batches
from torch.autograd import Variable

from util import init_model as init_convnet
from util import process_batches

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
device = torch.device("cuda:0")

available_features = ['nasnetalarge', 'resnet152', 'pnasnet5large', 'densenet121', 'senet154', 'polynet', 'vgg16']


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('ckpt_file', help='Path to the NeXtVLAD checkpoint file.')
    opt.add_argument('pca_dir', help='Directory containing PCA data.')
    opt.add_argument('files', nargs='+', help='List of files to process.')
    opt.add_argument('-gl', '--gpu_list',
                     required=True, nargs='+', type=int,
                     help="Space delimited list of GPU indices to use. Example for 4 GPUs: -gl 0 1 2 3")
    opt.add_argument('-bs', '--batch_size', type=int,
                     help="Batch size to use during feature extraction. Larger batch size = more VRAM usage",
                     default=8)
    opt.add_argument('--type', required=True,
                     help='ConvNet to use for processing features.',
                     choices=available_features)
    opt.add_argument('--max_frames', help="Max frames length of dataset.", default=50, type=int)
    opt.add_argument('--num_classes', help="Number of classes that was in train dataset.", default=5, type=int)

    opt = vars(opt.parse_args())

    logger.info("Found {} GPUs, using {}.".format(torch.cuda.device_count(), len(opt['gpu_list'])))

    # Convnet
    tf_img, convnet = init_convnet(opt['gpu_list'], opt['type'])
    # PCA
    eigenvecs = np.load(os.path.join(opt['pca_dir'], 'eigenvecss.npy'))
    eigenvals = np.load(os.path.join(opt['pca_dir'], 'eigenvals.npy'))
    center = np.load(os.path.join(opt['pca_dir'], 'mean.npy'))
    # neXtVLAD
    model = NeXtVLADModel(opt['num_classes'], max_frames=opt['max_frames'])
    model.load_state_dict(torch.load(opt['ckpt_file']))
    model.to(device)
    model.eval()

    for video in opt['files']:
        probe = ffmpeg.probe(video)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        out, _ = (
            ffmpeg
            .input(video)
            # .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=1)
            .run(capture_stdout=True)
        )
        video_np = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, height, width, 3])
        )
        batches = create_batches(video_np, tf_img, batch_size=opt['batch_size'])
        feats = process_batches(batches, opt['type'], opt['gpu_list'], convnet)

        fpca = np.zeros((len(feats), eigenvecs.T.shape[1]))
        for i, feat in enumerate(feats):
            fpca[i] = feature_pca(feat, center, eigenvals, eigenvecs)

        n = min(opt['max_frames'], len(fpca))
        padded = np.zeros((opt['max_frames'], fpca.shape[1]))
        padded[:n, :] = fpca[:n, :]
        mask = np.zeros((opt['max_frames'],))
        mask[:n] = 1

        fc_feats = Variable(torch.from_numpy(padded).type(torch.FloatTensor)).to(device)
        mask = Variable(torch.from_numpy(mask).type(torch.FloatTensor)).to(device)

        out = model(fc_feats.unsqueeze(0), mask=mask.unsqueeze(0))
        print(f"{video}: {out.argmax().detach().cpu().numpy()}")
