import glob
import json
import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class OTB(BaseVideoDataset):
    """ OTB dataset.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasot_dir if root is None else root
        root = '/mnt/data1/tzh/data/OTB_sentences/'
        super().__init__('OTB', root, image_loader)
        # Keep a list of all classes
        self.class_list = [f for f in os.listdir(self.root)]
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}
        self.embed_dc = np.load('/mnt/data1/tzh/Stark/otb_roberta_embed.npy', allow_pickle=True).item()
        self.sequence_list = self._build_sequence_list(vid_ids, split)
        self.cap_size = 20
        self.bert_emb_size = 768
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.seq_per_class = self._build_class_list()
        self.match_dc = self._build_match()

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'otb_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'otb_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        elif vid_ids is not None:
            sequence_list = [c + '-' + str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list

    def _build_match(self):
        match_dc = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            seq_path = self._get_sequence_path(seq_id)
            match_dc[seq_path] = sorted(glob.glob('{}/img/*'.format(seq_path)))
        return match_dc

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'otb'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = []
        for line in open(bb_anno_file, 'r').readlines():
            line = line.replace('\n', '')
            for sp in ['\t', ',', ' ']:
                if sp in line:
                    gt.append([int(k) for k in line.split(sp)])
                    continue
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]

        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        # print('get_sequence_info', seq_id)
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()

        caption = self.embed_dc[seq_path + '/language.txt'].copy()
        tokens_len = min(caption['last_hidden_state'].shape[1], self.cap_size)
        hidden = np.zeros((self.cap_size, self.bert_emb_size), dtype=np.float32)
        masks = np.ones(self.cap_size)

        masks[:tokens_len] = 0
        # print(caption['last_hidden_state'].shape, seq_path)
        hidden[:tokens_len] = caption['last_hidden_state'][0, :tokens_len]

        caption['last_hidden_state'] = hidden
        caption['masks'] = masks
        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'words': caption}

    def _get_frame_path(self, seq_path, frame_id):
        return self.match_dc[seq_path][frame_id]

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key == 'words':
                continue
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
