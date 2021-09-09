from transformers import RobertaTokenizer, RobertaModel
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import shutil, os
import sys
import pandas as pd

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = RobertaModel.from_pretrained('roberta-base').cuda()
seq_items = np.load('/mnt/data1/tzh/data/flickr/items.npy', allow_pickle=True)
st = 300000
ed = 372792
for i in tqdm(range(st, ed)):
    item = seq_items[i]
    if os.path.exists('/mnt/data1/tzh/Stark/refs/flickr/{}.npy'.format(i)):
        continue
    cap = item['nlp']
    inputs = tokenizer(cap, return_tensors="pt", padding=True)
    for k in inputs:
        inputs[k] = inputs[k].cuda()
    outputs = bert_model(**inputs)
    item = {
        'pool_out': outputs.pooler_output.detach().cpu().numpy(),
        'last_hidden_state': outputs.last_hidden_state.detach().cpu().numpy(),
    }
    np.save('/mnt/data1/tzh/Stark/refs/flickr/{}.npy'.format(i), item)
