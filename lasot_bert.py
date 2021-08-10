# from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import json
import numpy as np
from tqdm import tqdm

cap_dc = json.load(open('/mnt/data1/tzh/Stark/lasot_nlp.json'))
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = RobertaModel.from_pretrained('roberta-base')
dc = {}
for k in tqdm(cap_dc):
    cap = cap_dc[k]
    inputs = tokenizer(cap, return_tensors="pt", padding=True)
    outputs = bert_model(**inputs)
    dc[k] = {
        'pool_out': outputs.pooler_output.detach().cpu().numpy(),
        'last_hidden_state': outputs.last_hidden_state.detach().cpu().numpy(),
    }
np.save('/mnt/data1/tzh/Stark/lasot_reberta_embed.npy', dc)
