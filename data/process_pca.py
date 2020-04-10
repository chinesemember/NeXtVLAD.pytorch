import os
import argparse
import numpy as np

from tqdm import tqdm
from fnmatch import filter
from sklearn.decomposition import PCA

from util import feature_pca_whiten, feature_pca


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('training_features_folder', help="Folder containing full-scale training features.")
    opt.add_argument('test_features_folder', help="Folder containing full-scale test features.")
    opt.add_argument('save_folder', help="Folder to save PCA params and features.")
    opt = vars(opt.parse_args())

    D = []
    for root, dirs, filenames in os.walk(opt['training_features_folder']):
        for npf_name in filter(filenames, "*.npy"):
            npf_path = os.path.join(root, npf_name)
            npo = np.load(npf_path)
            D.extend(npo)

    print(f"Generating PCA vectors...")
    pca = PCA(n_components=1024)
    pca.fit(D)
    eigenvecs = pca.components_
    eigenvals = pca.explained_variance_
    center = pca.mean_
    np.save(os.path.join(opt['save_folder'], 'eigenvecss.npy'), eigenvecs)
    np.save(os.path.join(opt['save_folder'], 'eigenvals.npy'), eigenvals)
    np.save(os.path.join(opt['save_folder'], 'mean.npy'), center)

    for split_folder in [opt['training_features_folder'], opt['test_features_folder']]:
        if split_folder == opt['training_features_folder']:
            out_root = os.path.join(opt['save_folder'], 'train_PCA-1024')
        elif split_folder == opt['test_features_folder']:
            out_root = os.path.join(opt['save_folder'], 'test_PCA-1024')
        else:
            break

        print(f"Created {out_root}")
        class_dirs = os.listdir(split_folder)
        num_classes = len(class_dirs)
        for k, class_dir in enumerate(class_dirs):
            print(f"Class {k+1}/{num_classes}")
            class_feats_dir = os.path.join(split_folder, class_dir)
            class_out_dir = os.path.join(out_root, class_dir)
            if not os.path.isdir(class_out_dir):
                os.makedirs(class_out_dir)

            for npff in tqdm(os.listdir(class_feats_dir)):
                npf = os.path.join(class_feats_dir, npff)
                feats = np.load(npf)
                feats_pca = np.zeros((len(feats), 1024))
                for i, feat in enumerate(feats):
                    # TODO: toggle whitening on/off
                    # pcaw = feature_pca_whiten(feat, center, eigenvals, eigenvecs)
                    pcaw = feature_pca(feat, center, eigenvals, eigenvecs)
                    feats_pca[i] = pcaw

                feats_pca_path = os.path.join(class_out_dir, npff)
                np.save(feats_pca_path, feats_pca)
