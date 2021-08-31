"""
STARK-ST Model (Spatio-Temporal).
"""
from .backbone import build_backbone
from .transformer import build_transformer
from .head import build_box_head, MLP
from lib.models.stark.stark_s import STARKS
import torch


class STARKST(STARKS):
    """ This is the base class for Transformer Tracking """

    def __init__(self, backbone, transformer, box_head, cls_head=None, nlp_cls_head=None,
                 nlp_transformer=None, nlp_box_head=None, cfg=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__(backbone, transformer, box_head, cfg,
                         nlp_transformer=nlp_transformer, nlp_box_head=nlp_box_head)
        self.cls_head = cls_head
        self.nlp_cls_head = nlp_cls_head

    def forward(self, img=None, seq_dict=None, mode="backbone", run_box_head=False, run_cls_head=False, caption=None,
                only=None):
        if mode == "backbone":
            return self.forward_backbone(img)
        elif mode == "transformer":
            return self.forward_transformer(seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head,
                                            caption=caption, only=only)
        else:
            raise ValueError

    def forward_transformer(self, seq_dict, run_box_head=False, run_cls_head=False, caption=None, only=None):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        out, output_embed, nlp_output_embed = super().forward_transformer(seq_dict, run_box_head, run_cls_head, caption,
                                                                          only)

        # Forward the corner head
        if run_cls_head:
            if only != 'nlp':
                out.update({'pred_logits': torch.sigmoid(self.cls_head(output_embed)[-1])})
            if only != 'box':
                out.update({'nlp_pred_logits': torch.sigmoid(self.nlp_cls_head(nlp_output_embed)[-1])})

        return out, output_embed, nlp_output_embed


def build_starkst(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    transformer = build_transformer(cfg, caption=False)
    box_head = build_box_head(cfg)
    cls_head = MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, 1, cfg.MODEL.NLAYER_HEAD)

    nlp_transformer = build_transformer(cfg, caption=True)
    nlp_box_head = build_box_head(cfg)
    nlp_cls_head = MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, 1, cfg.MODEL.NLAYER_HEAD)
    model = STARKST(
        backbone,
        transformer,
        box_head,
        cls_head=cls_head,
        nlp_cls_head=nlp_cls_head,
        nlp_transformer=nlp_transformer,
        nlp_box_head=nlp_box_head,
        cfg=cfg,
    )

    return model
