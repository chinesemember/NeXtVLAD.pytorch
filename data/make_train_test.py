'''
Modified from code here: https://github.com/Yidadaa/Pytorch-Video-Classification
'''

import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
import argparse
import ffmpeg

# 数据集的默认位置
# default params
default_output_dir = os.path.dirname(os.path.abspath(__file__))
default_src_dir = os.path.join(default_output_dir, 'UCF')
default_test_size = 0.2


def split(src_dir=default_src_dir, output_dir=default_src_dir, size=default_test_size):
    # 设置默认参数
    # set defaults
    src_dir = default_src_dir if src_dir is None else src_dir
    output_dir = default_output_dir if output_dir is None else output_dir
    size = default_test_size if size is None else size

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 生成测试集和训练集目录
    # split into train and test
    for folder in ['train', 'test']:
        folder_path = os.path.join(output_dir, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print('Folder {} is created'.format(folder_path))

    # 划分测试集和训练集
    train_set = []
    test_set = []
    classes = os.listdir(src_dir)
    num_classes = len(classes)
    for class_index, classname in enumerate(classes):
        print(f"Current class:\t{class_index+1}")
        # 读取所有视频路径
        videos = os.listdir(os.path.join(src_dir, classname))
        # 打乱视频名称
        np.random.shuffle(videos)
        # 确定测试集划分点
        split_size = int(len(videos) * size)

        # 生成训练集和测试集的文件夹
        for i in range(2):
            part = ['train', 'test'][i]
            class_dir = os.path.join(output_dir, part, classname)
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

        jobs = []
        # 遍历每个视频，将每个视频的图像帧提取出来
        for i in range(len(videos)):
            video_path = os.path.join(src_dir, classname, videos[i])

            video_type = 'test' if i <= split_size else 'train'
            video_name = videos[i].rsplit('.')[0]

            img_dir = os.path.join(output_dir, video_type, classname, f'{video_name}')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            if len(os.listdir(img_dir)) > 0:
                continue

            img_path = os.path.join(output_dir, video_type, classname, f'{video_name}/%6d.jpg')
            jobs.append({'in': video_path, 'out': img_path})

            info = [classname, video_name, img_path]
            # 将视频帧信息保存起来
            if video_type == 'test':
                test_set.append(info)
            else:
                train_set.append(info)

        def subproc_call(job):
            try:
                # sample at 1fps: https://arxiv.org/pdf/1609.08675.pdf
                process = (
                    ffmpeg
                    .input(job['in'])
                    .output(job['out'], pattern_type='glob', framerate=1)
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
                out, err = process.communicate()
            except ffmpeg.Error as e:
                print(e)
                print(err)

        # subproc_call(jobs[0])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # wrap with list to run .map generator on execution
            _ = list(tqdm(executor.map(subproc_call, jobs), total=len(jobs)))

        # 将训练集和测试集数据保存到文件中，方便写dataloader
        datas = [train_set, test_set]
        names = ['train', 'test']
        for i in range(2):
            with open(output_dir + '/' + names[i] + '.csv', 'w') as f:
                f.write('\n'.join([','.join(line) for line in datas[i]]))


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 make_train_test.py -i path/to/UCF -o path/to/output -s 0.3')
    parser.add_argument('-i', '--src_dir', help='path to UCF datasets', default=default_src_dir)
    parser.add_argument('-o', '--output_dir', help='path to output', default=default_output_dir)
    parser.add_argument('-s', '--size', help='ratio of test sets', default=default_test_size)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    split(**vars(args))
