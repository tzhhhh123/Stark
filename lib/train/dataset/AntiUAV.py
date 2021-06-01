import json

import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict

from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings


class AntiUAV(BaseVideoDataset):
    """ AntiUAV dataset.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split='train'):
        """
        args:
            root        - The path to the TrackingNet folder, containing the training sets.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            set_ids (None) - List containing the ids of the TrackingNet sets to be used for training. If None, all the
                            sets (0 - 11) will be used.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        # print(root, env_settings().antiuav_dir)
        if split == 'test':
            self.split_dir = 'test-challenge'
        else:
            self.split_dir = 'trainval'
        root = env_settings().antiuav_dir if root is None else root
        super().__init__('AntiUAV', root, image_loader)

        # Keep a list of all videos. Sequence list is a list of tuples (set_id, video_name) containing the set_id and
        # video_name for each sequence

        self.sequence_list = json.load(open(os.path.join(root, 'split.bk.json')))[split]
        self.sequence_list = [s.split('/')[1] for s in self.sequence_list]
        # self.sequence_list = os.listdir(os.path.join(root, self.split_dir))

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.seq_to_class_map, self.seq_per_class = self._load_class_info()

        # we do not have the class_lists for the tracking net
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def _load_class_info(self):
        seq_per_class = {}
        seq_per_class['uav'] = self.sequence_list
        seq_to_class_map = {}
        for i, seq in enumerate(self.sequence_list):
            seq_to_class_map[seq] = 'uav'

        return seq_to_class_map, seq_per_class

    def get_name(self):
        return 'AntiUAV'

    def has_class_info(self):
        return True

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_id):
        vid_name = self.sequence_list[seq_id]
        bb_anno_file = os.path.join(self.root, self.split_dir, vid_name, "IR_label.json")
        # gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
        #                      low_memory=False).values
        gt = json.load(open(bb_anno_file, "r"))['gt_rect']
        for i in range(len(gt)):
            if len(gt[i]) == 0:
                gt[i] = [0, 0, 0, 0]
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(seq_id)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_id, frame_id):
        vid_name = self.sequence_list[seq_id]
        frame_path = os.path.join(self.root, 'frames', self.split_dir, vid_name, 'IR',
                                  "{:0>6d}".format(frame_id) + ".jpg")
        return self.image_loader(frame_path)

    def _get_class(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return self.seq_to_class_map[seq_name]

    def get_class_name(self, seq_id):
        obj_class = self._get_class(seq_id)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = [self._get_frame(seq_id, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        obj_class = self._get_class(seq_id)

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
