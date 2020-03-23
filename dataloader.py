import json

import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from multiprocessing import Pool
from multiprocessing import Queue
from collections import defaultdict


class CocoDataset(Dataset):

    def __init__(self, coco_labels):
        # python 3
        # super().__init__()
        super(CocoDataset, self).__init__()
        self.coco_labels = list(coco_labels['labels'].items())
        self.num_classes = coco_labels['num_classes']

    def __getitem__(self, ix):
        labels = torch.zeros(self.num_classes)
        image_id, labels_ids = self.coco_labels[ix]
        labels[labels_ids] = 1
        data = {}
        data['image_ids'] = image_id
        data['labels'] = labels
        return data

    def __len__(self):
        return len(self.coco_labels)


pool_queue = Queue()
work = []


def _threaded_sample_load(vid_id, fpath, n_frame_steps):
    fc_feat = load_and_subsample_feat(fpath, n_frame_steps)
    pool_queue.put((vid_id, fc_feat))


class VideoClassificationFolder:
    def __init__(self, feats_folder: str):
        """
        Init the video classification folder with the following tree structure:
            - [train|test]
            |- class 0
            |- - video 0
            |- - ...
            |- - video_{0i}
            |- ...
            |- ...
            |- class k
            |- - video 0
            |- - ...
            |- - video_{ki}

            $i$ is not guaranteed to be consistent between classes
        :param feats_folder: root directory where features are stored
        """
        self.class_to_feats_map = defaultdict(list)
        self.feats_dir = feats_folder
        self.num_classes = len(os.listdir(self.feats_dir))
        for c in os.listdir(self.feats_dir):
            self.class_to_feats_map[c] = [os.path.join(self.feats_dir, c, npf) for npf in
                                          os.listdir(os.path.join(self.feats_dir, c))]

    def flattened(self) -> dict:
        """
        :return: a flattened tree as a dict of idx: 2-tuple (feats_path, class_id) with deterministic ordering
        """
        l = {}
        i = 0
        for c in sorted(list(self.class_to_feats_map.keys())):
            for feats_path in sorted(self.class_to_feats_map[c]):
                l[i] = (feats_path, c)
                i += 1

        return l

    def __len__(self) -> int:
        return sum([len(self.class_to_feats_map[c]) for c in list(self.class_to_feats_map.keys())])


class VideoClassificationDataset(Dataset):

    # def get_vocab_size(self):
    #     return len(self.get_vocab())

    # def get_vocab(self):
    #     return self.ix_to_word

    # def get_seq_length(self):
    #     return self.seq_length

    def __init__(self, opt, mode):
        # python 3
        # super().__init__()
        super(VideoClassificationDataset, self).__init__()
        self.mode = mode  # to load train/val/test data
        self.feats_dir = opt['feats_dir']
        self.max_frames = opt['max_frames']
        self.tree = VideoClassificationFolder(self.feats_dir)
        self.num_classes = self.tree.num_classes
        self.n = len(self.tree)
        # self.n_frame_steps = opt['n_frame_steps']
        # load in the sequence data

        if self.mode != 'inference':
            print(f'load feats from {self.feats_dir}')
            # Memory cache for features
            print(f"Pre-cache {self.n} features in memory.")
            self._feat_cache = {}
            # pool = Pool(16)

            for idx, (fc_feat_path, c) in self.tree.flattened().items():
                try:
                    fc_feat, mask = load_and_subsample_feat(fc_feat_path, self.max_frames)
                    self._feat_cache[idx] = (fc_feat, mask, c)
                except:
                    print(f"{fc_feat_path} was not found")

        self.classes = sorted(list(self.tree.class_to_feats_map.keys()))
        self.tree = self.tree.flattened()
        print("Finished initializing dataloader.")

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = ix % self.n

        fc_feat = self._feat_cache.get(ix, None)
        if fc_feat is None:
            fc_feat_path, c = self.tree[ix]
            fc_feat, mask = load_and_subsample_feat(fc_feat_path, self.max_frames)
            self._feat_cache[ix] = (fc_feat, mask, c)
        else:
            fc_feat, mask, c = self._feat_cache[ix]

        label = self.classes.index(c)

        data = {
            'fc_feats':  Variable(torch.from_numpy(fc_feat).type(torch.FloatTensor)),
            'ground_truth': Variable(torch.from_numpy(one_hot(label, self.num_classes)).type(torch.FloatTensor)),
            'video_id': ix,
            'mask': Variable(torch.from_numpy(mask).type(torch.FloatTensor))
        }
        return data

    def __len__(self):
        return self.n


def load_and_subsample_feat(fc_feat_path, max_frames, n_frame_steps=1):
    # fc_feat = np.load(fc_feat_path)
    # Subsampling
    # samples = np.round(np.linspace(
    #     0, fc_feat.shape[0] - 1, n_frame_steps)).astype(np.int32)
    try:
        fc_feat = np.load(fc_feat_path)
        padded = np.zeros((max_frames, fc_feat.shape[1]))
        padded[:len(fc_feat), :] = fc_feat
        mask = np.zeros((max_frames,))
        mask[:len(fc_feat)] = 1
    except Exception as e:
        print("Bad feature file in dataset: {}. Purge, re-process, and try again.".format(fc_feat_path))
        raise e
    return padded, mask


def one_hot(idx, num_classes):
    out = np.zeros(num_classes)
    out[idx] = 1
    return out


if __name__ == '__main__':
    opt = {
        'feats_dir': "/home/wgar/NeXtVLAD.pytorch/data/UCF101_debug/train_PCA-1024",
        'max_frames': 50
    }

    vd = VideoClassificationDataset(opt, 'train')
    data = vd.__getitem__(5)
    fc_feats = data['fc_feats']
    print(fc_feats.shape)
    print(data['mask'])