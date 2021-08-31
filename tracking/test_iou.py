import os
import sys
import argparse
import numpy as np
import cv2 as cv
from tqdm import tqdm
import torch

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset


def calc_iou_overlap(pred_bb, anno_bb):
    pred_bb = torch.Tensor(pred_bb)
    anno_bb = torch.Tensor(anno_bb)
    tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

    return intersection / union


def main():
    parser = argparse.ArgumentParser(description='visualization.')
    # data_dir = '/mnt/data1/tzh/data/LaSOT/LaSOTBenchmark'
    # data_dir = '/mnt/data1/tzh/data/OTB_sentences'
    root_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_st'
    save_root = '/mnt/data1/tzh/Stark/vis'
    parser.add_argument('--name', type=str, default='baseline_lasot',
                        help='lasot result path')
    parser.add_argument('--dataset', type=str, default='otb',
                        help='lasot result path')
    # parser.add_argument('--nlp', action='store_true', default=False, )
    args = parser.parse_args()
    root = os.path.join(root_dir, args.name)
    save_dir = os.path.join(save_root, args.name)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    # dataset = get_dataset('lasot')
    dataset = get_dataset(args.dataset)
    dc = {
        'box': [],
        'nlp': [],
        'fuse': {},
    }
    for seq in tqdm(dataset):
        try:
            name = seq.name
            txt_path = os.path.join(root, '{}.txt'.format(name))
            conf_path = os.path.join(root, '{}_logit.txt'.format(name))

            # nlp_path = os.path.join(root[:-5], '{}.txt'.format(name))
            # nlp_conf_path = os.path.join(root[:-5], '{}_nlp_logit.txt'.format(name))
            nlp_path = os.path.join(root, '{}_nlp.txt'.format(name))
            nlp_conf_path = os.path.join(root, '{}_nlp_logit.txt'.format(name))

            # score_path = os.path.join(root, '{}.txt'.format(name))
            if os.path.exists(txt_path) is False:
                print('miss' + name)
                continue

            out_res = np.loadtxt(open(txt_path, 'r'), dtype=float)[1:]
            nlp_res = np.loadtxt(open(nlp_path, 'r'), dtype=float)
            conf = np.loadtxt(open(conf_path, 'r'), dtype=float)
            # nlp_conf = np.loadtxt(open(nlp_conf_path, 'r'), dtype=float)
            gt = np.array(seq.ground_truth_rect[1:])

            only_bbox = calc_iou_overlap(out_res, gt)
            only_nlp = calc_iou_overlap(nlp_res, gt)
            for th in range(1, 20):
                th0 = th / 20
                fuse = []
                for i in range(len(gt)):
                    if conf[i] < th0:
                        fuse.append(nlp_res[i])
                    else:
                        fuse.append(out_res[i])
                fuse_res = calc_iou_overlap(fuse, gt)
                if th0 not in dc['fuse']:
                    dc['fuse'][th0] = []
                dc['fuse'][th0].append(fuse_res.mean())
            dc['box'].append(only_bbox.mean())
            dc['nlp'].append(only_nlp.mean())
            # argx = only_bbox + 0.3 < only_nlp
            # argx = conf < 0.3
            # print(conf[argx])
            # print(only_bbox[argx])
            # print(only_nlp[argx])
            # print(nlp_conf[argx])
            # import ipdb
            # ipdb.set_trace()
        except Exception as e:
            print(e)
            import traceback
            print(traceback.format_exc())
            print(seq.name)
        # import ipdb
        # ipdb.set_trace()

        # print("bbox:{}\n nlp:{}\n fuse:{}".format(only_bbox, only_nlp, fuse_res))
    print('box:{}   nlp:{}  '.format(sum(dc['box']), sum(dc['nlp'])))
    for th in dc['fuse']:
        print("th: {}  iou: {}".format(th, sum(dc['fuse'][th])))
    # import ipdb
    # ipdb.set_trace()


if __name__ == '__main__':
    main()
