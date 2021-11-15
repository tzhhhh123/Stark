"""
Basic STARK Model (Spatial-only).
"""
import torch
from torch import nn

from lib.utils.misc import NestedTensor

from .backbone import build_backbone
from .transformer import build_transformer
from .head import build_box_head, MLP
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.merge import merge_template_search


class STARKS(nn.Module):
    """ This is the base class for Transformer Tracking """

    def __init__(self, backbone, transformer, box_head, cfg, nlp_transformer=None, nlp_box_head=None, fuse_head=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.nlp_transformer = nlp_transformer
        self.box_head = box_head
        self.nlp_box_head = nlp_box_head
        self.aux_loss = cfg.TRAIN.DEEP_SUPERVISION
        self.head_type = cfg.MODEL.HEAD_TYPE
        self.token_size = cfg.MODEL.TOKEN_SIZE

        self.num_queries = cfg.MODEL.NUM_OBJECT_QUERIES
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)  # object queries
        self.nlp_query_embed = nn.Embedding(self.num_queries, hidden_dim)  # object queries
        self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer

        # to fix
        self.fuse_type = cfg.MODEL.FUSE_TYPE
        self.fuse_head = fuse_head
        if self.fuse_type == "CORNER":
            self.feat_sz_s = int(fuse_head.feat_sz)
            self.feat_len_s = int(fuse_head.feat_sz ** 2)

        if self.head_type == "CORNER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, img=None, seq_dict=None, mode="backbone", run_box_head=True, run_cls_head=False, caption=None):

        if mode == "backbone":
            return self.forward_backbone(img)
        elif mode == "transformer":
            return self.forward_transformer(seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head,
                                            caption=caption)
        else:
            raise ValueError

    def forward_backbone(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust(output_back, pos)

    def forward_transformer(self, seq_dict, run_box_head=True, run_cls_head=False, caption=None, only=None):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        assert only in ['nlp', 'box', None]
        nlp_output_embed, nlp_enc_mem = None, None
        out = {}
        if only == 'nlp' or only is None:
            nlp_output_embed, nlp_enc_mem = self.nlp_transformer(seq_dict[-1]["feat"], seq_dict[-1]["mask"],
                                                                 self.nlp_query_embed.weight,
                                                                 # torch.zeros(self.num_queries, 256).cuda(),
                                                                 seq_dict[-1]["pos"], return_encoder_output=True,
                                                                 caption=caption)
            if run_box_head:
                nlp_out, nlp_outputs_coord = self.forward_box_head(nlp_output_embed, nlp_enc_mem, self.nlp_box_head)
                out['nlp_pred_boxes'] = nlp_out['pred_boxes']

            if only == 'nlp':
                return out, None, nlp_output_embed

        ###box way
        seq_dict = merge_template_search(seq_dict)

        box_output_embed, box_enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"],
                                                         self.query_embed.weight,
                                                         seq_dict["pos"], return_encoder_output=True,
                                                         caption=None)

        # Forward the corner head
        if run_box_head:
            box_out, outputs_coord = self.forward_box_head(box_output_embed, box_enc_mem, self.box_head)
            out['pred_boxes'] = box_out['pred_boxes']
        if self.fuse_type != "":
            fuse_out, _ = self.forward_fuse_head(box_output_embed, box_enc_mem, nlp_output_embed, nlp_enc_mem,
                                                 self.fuse_head)
            out['fuse_pred_boxes'] = fuse_out['pred_boxes']

        return out, box_output_embed, nlp_output_embed

    def att_d2c(self, hs, memory):
        enc_opt = memory[-self.feat_len_s:].transpose(0, 1)
        # encoder output for the search region (B, HW, C)
        dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
        att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, Nq * C, self.feat_sz_s, self.feat_sz_s)  ## -1 = bs
        return opt_feat

    def forward_fuse_head(self, hs, memory, nlp_hs, nlp_memory, fuse_head):
        if self.fuse_type == "MLP":
            outputs_coord = fuse_head(torch.cat((hs, nlp_hs), dim=-1)).sigmoid()
            out = {'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out, outputs_coord
        else:
            _, bs, nq, c = hs.size()

            # opt_feat = self.att_d2c(hs, memory)
            #
            # nlp_opt_feat = self.att_d2c(nlp_hs, nlp_memory)
            opt_feat = memory[-self.feat_len_s:].transpose(0, 1)
            # encoder output for the search region (B, HW, C)

            nlp_opt_feat = nlp_memory[-self.feat_len_s:].transpose(0, 1)

            outputs_coord = fuse_head(opt_feat, nlp_opt_feat, hs, nlp_hs).sigmoid()

            out = {'pred_boxes': outputs_coord}
            return out, outputs_coord

    def forward_box_head(self, hs, memory, box_head):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if self.head_type == "CORNER":
            # adjust shape
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1)
            # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
                (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, Nq * C, self.feat_sz_s, self.feat_sz_s)  ## -1 = bs
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(box_head(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "MLP":
            # Forward the class and box head
            outputs_coord = box_head(hs).sigmoid()
            out = {'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out, outputs_coord

    def adjust(self, output_back: list, pos_embed: list):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]


def build_starks(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    transformer = build_transformer(cfg, caption=False)
    box_head = build_box_head(cfg)
    fuse_head = MLP(cfg.MODEL.HIDDEN_DIM * 2, cfg.MODEL.HIDDEN_DIM, 4, 3)
    nlp_transformer = build_transformer(cfg, caption=True)
    nlp_box_head = build_box_head(cfg)

    model = STARKS(
        backbone,
        transformer,
        box_head,
        cfg=cfg,
        nlp_transformer=nlp_transformer,
        nlp_box_head=nlp_box_head,
        fuse_head=fuse_head,
    )

    return model
