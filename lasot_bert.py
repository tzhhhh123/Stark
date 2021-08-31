from transformers import RobertaTokenizer, RobertaModel
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import shutil, os
import sys
import argparse

parser = argparse.ArgumentParser(description='Choosed datasets')
parser.add_argument('dataset', type=str, help='Name of tracking method.')
args = parser.parse_args()
###REFCOCO*
sys.path.append('/mnt/data1/tzh/Stark')
sys.path.append('/mnt/data1/tzh/Stark/refer')
from refer import REFER

dataset = args.dataset
if dataset == 'refcocog':
    refer = REFER('/mnt/data1/tzh/data/refer', dataset=dataset, splitBy='umd')
else:
    refer = REFER('/mnt/data1/tzh/data/refer', dataset=dataset, splitBy='unc')
cap_dc = {}
ref_ids = refer.getRefIds()
refs = refer.loadRefs(ref_ids)

for ref in refs:
    for sen_id, s in zip(ref['sent_ids'], ref['sentences']):
        sen = ' '
        sen = sen.join(s['tokens'])
        cap_dc[sen_id] = sen
# cap_dc = json.load(open('/mnt/data1/tzh/Stark/lasot_nlp.json'))
print(len(cap_dc))
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = RobertaModel.from_pretrained('roberta-base').cuda()
dc = {}
for k in tqdm(cap_dc):
    cap = cap_dc[k]
    inputs = tokenizer(cap, return_tensors="pt", padding=True)
    for t in inputs:
        inputs[t] = inputs[t].cuda()
    outputs = bert_model(**inputs)
    dc[k] = {
        'pool_out': outputs.pooler_output.detach().cpu().numpy(),
        'last_hidden_state': outputs.last_hidden_state.detach().cpu().numpy(),
    }
np.save('/mnt/data1/tzh/Stark/{}_roberta_embed.npy'.format(dataset), dc)
