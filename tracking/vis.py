import os
import sys
import argparse
import numpy as np
import cv2 as cv
from tqdm import tqdm

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset


def frame2video(frames, save_path, dec=1):
    writer = None
    save_path = save_path.replace('.mkv', '.mp4')
    save_path = save_path.replace('.webm', '.mp4')
    for img in frames:
        if writer is None:
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            writer = cv.VideoWriter(save_path, fourcc, 25 // dec,
                                    (img.shape[1], img.shape[0]))
        if writer is not None:
            writer.write(img)


def main():
    parser = argparse.ArgumentParser(description='visualization.')
    # data_dir = '/mnt/data1/tzh/data/LaSOT/LaSOTBenchmark'
    data_dir = '/mnt/data1/tzh/data/OTB_sentences'
    root_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_s'
    save_root = '/mnt/data1/tzh/Stark/vis'
    parser.add_argument('--name', type=str, default='baseline_lasot',
                        help='lasot result path')
    args = parser.parse_args()
    root = os.path.join(root_dir, args.name)
    save_dir = os.path.join(save_root, args.name)
    # dataset = get_dataset('lasot')
    dataset = get_dataset('otb')
    for seq in tqdm(dataset):
        name = seq.name
        txt_path = os.path.join(root, '{}.txt'.format(name))
        # score_path = os.path.join(root, '{}.txt'.format(name))
        if os.path.exists(txt_path) is False:
            print('miss' + name)
            continue
        out_res = np.loadtxt(open(txt_path, 'r'), dtype=float).tolist()
        # out_score = np.loadtxt(open(txt_path, 'r'), dtype=float).tolist()
        caption = list(open(os.path.join(data_dir, name, 'language.txt')).readlines())[0]
        gt = seq.ground_truth_rect
        v_images = []
        for i, frame_path in enumerate(seq.frames):
            if i >= len(gt):
                continue
            image = cv.imread(frame_path)
            out_box = out_res[i]
            _gt = gt[i]
            cv.rectangle(image, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                         (0, 255, 0))
            cv.putText(image, caption, (20, 20), 1, 1.2,
                       (0, 0, 255), 1)
            # cv.putText(image, str(out['conf_score'])[:4], (int(out_box[0]), int(out_box[1])), 1, 2,
            #            (0, 0, 255), 2)
            cv.rectangle(image, (int(out_box[0]), int(out_box[1])),
                         (int(out_box[0] + out_box[2]), int(out_box[1] + out_box[3])),
                         (0, 255, 255))
            v_images.append(image)
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        frame2video(v_images, '{}/{}.mp4'.format(save_dir, seq.name))


if __name__ == '__main__':
    main()
