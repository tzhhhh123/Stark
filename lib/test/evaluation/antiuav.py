import json

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class antiuavDataset(BaseDataset):
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
        self.base_path = self.env_settings.antiuav
        if split == 'test':
            self.split_name = 'test-challenge'
        elif split == 'val':
            self.split_name = 'test-challenge'
        else:
            self.split_name = 'test-dev'

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = os.path.join(self.base_path, self.split_name, sequence_name, 'IR_label.json')
        ground_truth_rect = json.load(open(anno_path, 'r'))['gt_rect'][0]

        frames_path = '{}/{}'.format(self.base_path, 'frames', self.split_name, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'antiuav', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        sequence_list = os.listdir(os.path.join(self.base_path, self.split_name))
        return sequence_list
