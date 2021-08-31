import os
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import torch
import random
import numpy as np
from collections import OrderedDict
from lib.train.admin import env_settings
from refer import REFER


class REFERSeq(BaseVideoDataset):
    """ The refcocoXX dataset.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", dataname='refcoco'):
        """
        args:
            root - path to the coco dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            data_fraction (None) - Fraction of images to be used. The images are selected randomly. If None, all the
                                  images  will be used
            split - 'train' or 'val'.
            version - version of coco dataset (2014 or 2017)
        """
        root = '/mnt/data1/tzh/data/refer/'
        self.img_pth = os.path.join(root, 'images/mscoco/images/train2014')
        super().__init__('REFER', root, image_loader)
        if dataname == 'refcoco':
            self.refer = REFER(root, dataset='refcoco', splitBy='unc')
        elif dataname == 'refcoco+':
            self.refer = REFER(root, dataset='refcoco+', splitBy='unc')
        elif dataname == 'refcocog':
            self.refer = REFER(root, dataset='refcocog', splitBy='umd')
        elif dataname == 'refclef':  ####to fix
            self.refer = REFER(root, dataset='refclef', splitBy='unc')
            self.img_pth = os.path.join(root, 'images/saiapr_tc-12')
        else:
            raise ValueError('RERFER ERROR')

        self.ref_ids = self.refer.getRefIds(split=split)
        self.sen2ref = {}
        self.cap_size = 30
        self.bert_emb_size = 768
        self.embed_dc = np.load('/mnt/data1/tzh/Stark/{}_roberta_embed.npy'.format(dataname), allow_pickle=True).item()
        self.sequence_list = self._get_sequence_list()
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        # self.seq_per_class = self._build_seq_per_class()

    def _get_sequence_list(self):
        seq_list = []
        for ref_id in self.ref_ids:
            ref = self.refer.loadRefs(ref_id)[0]
            for sent_id in ref['sent_ids']:
                seq_list.append(sent_id)
                self.sen2ref[sent_id] = ref['ref_id']
        return seq_list

    def is_video_sequence(self):
        return False

    def get_name(self):
        return 'refer'

    def has_class_info(self):
        return False

    def has_segmentation_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        seq_id = self.sequence_list[seq_id]
        bbox = torch.Tensor(self.refer.getRefBox(self.sen2ref[seq_id])).long().view(1, 4)

        '''2021.1.3 To avoid too small bounding boxes. Here we change the threshold to 50 pixels'''
        valid = (bbox[:, 2] > 50) & (bbox[:, 3] > 50)

        visible = valid.clone().byte()

        caption = self.embed_dc[seq_id].copy()
        tokens_len = min(caption['last_hidden_state'].shape[1], self.cap_size)
        hidden = np.zeros((self.cap_size, self.bert_emb_size), dtype=np.float32)
        masks = np.ones(self.cap_size)

        masks[:tokens_len] = 0
        hidden[:tokens_len] = caption['last_hidden_state'][0, :tokens_len]

        caption['last_hidden_state'] = hidden
        caption['masks'] = masks

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'words': caption}

    def _get_frames(self, seq_id):
        ref = self.refer.loadRefs(self.sen2ref[seq_id])[0]
        path = self.refer.loadImgs(ref['image_id'])[0]['file_name']
        img = self.image_loader(os.path.join(self.img_pth, path))
        return img

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        seq_id = self.sequence_list[seq_id]
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key == 'words':
                continue
            anno_frames[key] = [value[0, ...] for _ in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
