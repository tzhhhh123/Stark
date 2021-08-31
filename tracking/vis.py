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
    root_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_st'
    save_root = '/mnt/data1/tzh/Stark/vis'
    parser.add_argument('--name', type=str, default='baseline_lasot',
                        help='lasot result path')
    # parser.add_argument('--nlp', action='store_true', default=False, )
    args = parser.parse_args()
    root = os.path.join(root_dir, args.name)
    save_dir = os.path.join(save_root, args.name)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    # dataset = get_dataset('lasot')
    dataset = get_dataset('otb')
    for seq in tqdm(dataset):
        name = seq.name
        txt_path = os.path.join(root, '{}.txt'.format(name))
        nlp_path = os.path.join(root, '{}_nlp.txt'.format(name))
        conf_path = os.path.join(root, '{}_logit.txt'.format(name))
        nlp_conf_path = os.path.join(root, '{}_nlp_logit.txt'.format(name))
        # score_path = os.path.join(root, '{}.txt'.format(name))
        if os.path.exists(txt_path) is False:
            print('miss' + name)
            continue

        out_res = np.loadtxt(open(txt_path, 'r'), dtype=float).tolist()[1:]
        # nlp_res = np.loadtxt(open(nlp_path, 'r'), dtype=float).tolist()
        # conf = np.loadtxt(open(conf_path, 'r'), dtype=float).tolist()
        # nlp_conf = np.loadtxt(open(nlp_conf_path, 'r'), dtype=float).tolist()
        # out_score = np.loadtxt(open(txt_path, 'r'), dtype=float).tolist()
        # import ipdb
        # ipdb.set_trace()
        caption = list(open(os.path.join(data_dir, name, 'language.txt')).readlines())[0]
        gt = seq.ground_truth_rect
        v_images = []
        for i, frame_path in enumerate(seq.frames):
            if i >= len(gt):
                continue
            image = cv.imread(frame_path)
            _gt = gt[i]
            cv.putText(image, caption, (20, 20), 1, 1.2, (0, 0, 255), 1)
            cv.rectangle(image, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                         (0, 255, 0))
            if i > 0:
                out_box = out_res[i - 1]
                # nlp_out_box = nlp_res[i - 1]

                # cv.putText(image, "{:.4f}".format(conf[i - 1]), (int(out_box[0]), int(out_box[1])), 1, 1,
                #            (0, 0, 255), 2)
                cv.rectangle(image, (int(out_box[0]), int(out_box[1])),
                             (int(out_box[0] + out_box[2]), int(out_box[1] + out_box[3])),
                             (0, 255, 255))

                # cv.putText(image, "{:.4f}".format(nlp_conf[i - 1]), (int(nlp_out_box[0]), int(nlp_out_box[1])), 1, 1,
                #            (0, 0, 255), 2)
                # cv.rectangle(image, (int(nlp_out_box[0]), int(nlp_out_box[1])),
                #              (int(nlp_out_box[0] + nlp_out_box[2]), int(nlp_out_box[1] + nlp_out_box[3])),
                #              (255, 255, 255))
            v_images.append(image)

        frame2video(v_images, '{}/{}.mp4'.format(save_dir, seq.name))


if __name__ == '__main__':
    main()
