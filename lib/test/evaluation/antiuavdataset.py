import json

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class AntiUAVDataset(BaseDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, split):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = self.env_settings.antiuav_path
        if split == 'test':
            # self.split_name = 'test-challenge'
            self.split_name = 'trainval'
        elif split == 'val':
            # self.split_name = 'test-challenge'
            self.split_name = 'trainval'
        else:
            self.split_name = 'test-challenge'

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = os.path.join(self.base_path, self.split_name, sequence_name, 'IR_label.json')
        gt = json.load(open(anno_path, 'r'))['gt_rect']
        target_visible = []
        for i in range(len(gt)):
            if len(gt[i]) == 0:
                target_visible.append(0)
                gt[i] = [0, 0, 0, 0]
            else:
                target_visible.append(1)
        target_visible = np.array(target_visible)
        ground_truth_rect = np.array(gt)
        # print(ground_truth_rect.shape)
        frames_path = os.path.join(self.base_path, 'frames', self.split_name, sequence_name, "IR")
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'antiuav', ground_truth_rect.reshape(-1, 4),
                        target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        # sequence_list = os.listdir(os.path.join(self.base_path, self.split_name))
        # sequence_list = np.loadtxt(os.path.join(self.base_path, 'val_40.txt'), dtype=str)
        # sequence_list = [s.replace('trainval/', '') for s in sequence_list]
        sequence_list = json.load(open(os.path.join(self.base_path, 'split.bk.json')))[split]
        sequence_list = [s.split('/')[1] for s in sequence_list]
        return sequence_list
