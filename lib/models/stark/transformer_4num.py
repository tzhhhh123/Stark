"""
STARK Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers

2020.12.23 Split some preprocess fom the forward function
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def check_inf(tensor):
    return torch.isinf(tensor.detach()).any()


def check_nan(tensor):
    return torch.isnan(tensor.detach()).any()


def check_valid(tensor, type_name):
    if check_inf(tensor):
        print("%s is inf." % type_name)
    if check_nan(tensor):
        print("%s is nan" % type_name)


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_obj_query=1,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, divide_norm=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        decoder_norm = nn.LayerNorm(d_model)
        if num_decoder_layers == 0:
            self.decoder = None
        else:
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

        # position_embeddings for caption
        self.cap_size = 30
        self.type_size = 2
        self.position_embeddings = nn.Embedding(
            self.cap_size, d_model
        )
        # token_type_embeddings for img and caption
        self.token_type_embeddings = nn.Embedding(
            self.type_size, d_model
        )
        self.bert_fc = nn.Linear(768, d_model)

        # split the text query into N
        self.soft_fc = nn.Linear(d_model, num_obj_query)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat, mask, query_embed, pos_embed, mode="all", return_encoder_output=False, caption=None):
        """

        :param feat: (H1W1+H2W2, bs, C)
        :param mask: (bs, H1W1+H2W2)
        :param query_embed: (N, C) or (N, bs, C)
        :param pos_embed: (H1W1+H2W2, bs, C)

        :param caption:(
            words_emd: (L,bs,C1)
            words_mask: (bs, L)
            words_pool: (1, bs, C1)
        )
        :param mode: run the whole transformer or encoder only
        :param return_encoder_output: whether to return the output of encoder (together with decoder)



        :return:
        """
        assert mode in ["all", "encoder"]
        bs = feat.size(1)
        Nq = query_embed.size(0)
        if caption is not None:
            words_emd, words_mask, words_pool = caption
            img_len = feat.shape[0]
            words_len = words_emd.shape[0]
            ## make segment_embedding for both img and words
            segment_ids = torch.tensor([1] * words_len + [0] * img_len).to(feat.device)  ###(LL = words_len + img_len)
            segment_emd = self.token_type_embeddings(segment_ids).unsqueeze(1).repeat(1, bs, 1)  ##(LL, bs, C)
            mask = torch.cat((words_mask.to(mask), mask), dim=1)  ##(bs,L)

            ## make extra position_embeding for words
            position_ids = torch.arange(
                words_len, dtype=torch.long, device=feat.device
            )
            bert_pos = self.position_embeddings(position_ids).unsqueeze(1).repeat(1, bs, 1)  ###(L,bs,C)
            pos_embed = torch.cat((bert_pos, pos_embed), dim=0)

            ## all then add to the emd
            pos_embed = pos_embed + segment_emd

            ##make words_feat
            words_feat = self.bert_fc(words_emd)  ##(30,bs,C)
            feat = torch.cat((words_feat, feat), dim=0)  ##(30+64+400,bs,C)

            # import ipdb
            # ipdb.set_trace()
            soft_logit = self.soft_fc(words_feat).permute(1, 2, 0) * \
                         (1 - words_mask).float().unsqueeze(1).repeat(1, Nq, 1)  ###(bs,N,30)

            soft = torch.softmax(soft_logit, dim=2)
            words_pool = torch.matmul(soft, words_feat.permute(1, 0, 2)).transpose(
                0, 1)  ##(B,N,30) x (B,30,C) --> (B,N,C)  -->(N,B,C)
            # import ipdb
            # ipdb.set_trace()

        if self.encoder is None:
            memory = feat
        else:
            memory = self.encoder(feat, src_key_padding_mask=mask, pos=pos_embed)
        if mode == "encoder":
            return memory
        elif mode == "all":
            assert len(query_embed.size()) in [2]  ###[2,3]
            if len(query_embed.size()) == 2:
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (N,C) --> (N,1,C) --> (N,B,C)

            if caption is not None:
                query_embed = query_embed + words_pool  ###only add

            if self.decoder is not None:
                tgt = torch.zeros_like(query_embed)
                hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos=query_embed)
            else:
                hs = query_embed.unsqueeze(0)
            if return_encoder_output:
                return hs.transpose(1, 2), memory  # (1, B, N, C)
            else:
                return hs.transpose(1, 2)  # (1, B, N, C)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # first normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # add pos to src
        if self.divide_norm:
            # print("encoder divide by norm")
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = k / torch.norm(k, dim=-1, keepdim=True)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # self-attention
        q = k = self.with_pos_embed(tgt, query_pos)  # Add object query to the query and key
        if self.divide_norm:
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = k / torch.norm(k, dim=-1, keepdim=True)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # mutual attention
        queries, keys = self.with_pos_embed(tgt, query_pos), self.with_pos_embed(memory, pos)
        if self.divide_norm:
            queries = queries / torch.norm(queries, dim=-1, keepdim=True) * self.scale_factor
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)
        tgt2 = self.multihead_attn(query=queries,
                                   key=keys,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(cfg):
    return Transformer(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
        nhead=cfg.MODEL.TRANSFORMER.NHEADS,
        dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.MODEL.TRANSFORMER.ENC_LAYERS,
        num_decoder_layers=cfg.MODEL.TRANSFORMER.DEC_LAYERS,
        num_obj_query=cfg.MODEL.NUM_OBJECT_QUERIES,
        normalize_before=cfg.MODEL.TRANSFORMER.PRE_NORM,
        return_intermediate_dec=False,  # we use false to avoid DDP error,
        divide_norm=cfg.MODEL.TRANSFORMER.DIVIDE_NORM,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")