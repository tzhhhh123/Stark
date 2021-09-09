import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import pandas
import glob


class TNL2KDataset(BaseDataset):
    """
    TNL2K test set consisting of 700 videos
    """

    def __init__(self):
        super().__init__()
        self.base_path = '/mnt/data1/tzh/data/TNL2K_test_subset'
        self.sequence_list = self._get_sequence_list()
        self.embed_dc = np.load('/mnt/data1/tzh/Stark/tnl2k_roberta_embed.npy', allow_pickle=True).item()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        valid = (ground_truth_rect[:, 2] > 0) & (ground_truth_rect[:, 3] > 0)
        target_visible = valid

        frames_list = sorted(glob.glob('{}/{}/imgs/*'.format(self.base_path, sequence_name)))

        target_class = 0
        ###add caption
        cap_size = 20
        bert_emb_size = 768
        seq_path = '{}/{}'.format(self.base_path, sequence_name)
        caption = self.embed_dc[seq_path + '/language.txt']
        tokens_len = min(caption['last_hidden_state'].shape[1], cap_size)
        hidden = np.zeros((cap_size, bert_emb_size), dtype=np.float32)
        masks = np.ones(cap_size)

        masks[:tokens_len] = 0
        hidden[:tokens_len] = caption['last_hidden_state'][0, :tokens_len]

        caption['last_hidden_state'] = hidden
        caption['masks'] = masks

        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible, caption=caption)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        file_path = '/mnt/data1/tzh/Stark/lib/train/data_specs/tnl2k_val_split.txt'
        sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        return sequence_list
