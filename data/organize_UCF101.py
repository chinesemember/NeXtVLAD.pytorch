import os
import shutil
import fnmatch
import re
from tqdm import tqdm

if __name__ == '__main__':
    ind_filepath = "/mnt/nfs/hgst-raid1/WD-Passport_4TB/dataset/UCF101/ucfTrainTestlist/classInd.txt"
    vids_dir = "/mnt/nfs/hgst-raid1/WD-Passport_4TB/dataset/UCF101/videos"

    with open(ind_filepath, 'r') as f:
        lines = f.readlines()
        classes = [l.strip().split(' ')[1] for l in lines]
        for c in tqdm(classes):
            c_dir = os.path.join(vids_dir, c)
            if not os.path.exists(c_dir):
                os.makedirs(c_dir)
            rx = re.compile(fnmatch.translate(f"*{c}*.avi"), re.IGNORECASE)
            class_videos = list(filter(rx.search, os.listdir(vids_dir)))
            if len(class_videos) != 0:
                # script was already ran
                # continue
                from_paths = [os.path.join(vids_dir, cv) for cv in sorted(class_videos)]
                to_paths = [os.path.join(c_dir, cv) for cv in sorted(class_videos)]
                for from_path, to_path in zip(from_paths, to_paths):
                    shutil.move(from_path, to_path)
