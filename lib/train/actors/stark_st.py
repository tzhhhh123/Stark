from . import STARKSActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.box_ops import box_iou
import torch.nn as nn
import torch.nn.functional as F
import torch


class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean', ):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = logits
        # probs = torch.sigmoid(logits) ## sigmoid before
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                                F.softplus(logits, -1, 50),
                                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                                  -logits + F.softplus(logits, -1, 50),
                                  -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class STARKSTActor(STARKSActor):
    """ Actor for training the STARK-ST(Stage2)"""

    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective, loss_weight, settings)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.loss_type = settings.cls_type
        if self.loss_type == 'Focal':
            self.cls_loss = FocalLossV1()
        elif self.loss_type == 'BCE':
            self.cls_loss = torch.nn.BCELoss()
        else:
            self.cls_loss = torch.nn.SmoothL1Loss()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=True)

        # process the groundtruth label
        # gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)
        labels = data['label'].view(-1)  # (batch, ) 0 or 1

        loss, status = self.compute_losses(out_dict, labels)

        return loss, status

    def compute_losses(self, pred_dict, labels, return_status=True, only_cls=True):
        iou_pred = pred_dict["pred_logits"].view(-1)
        nlp_pred = pred_dict["nlp_pred_logits"].view(-1)
        Bloss = self.cls_loss(iou_pred, labels)
        Lloss = self.cls_loss(nlp_pred, labels)
        import random
        if random.random() > 0.995:
            print("\npred\n", iou_pred)
            print("\nnlp\n", nlp_pred)
            print("\nlabels\n", labels)

        loss = Bloss + Lloss
        # loss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), labels)
        if return_status:
            # status for log
            status = {
                "ALoss": loss.item(),
                "Bcls_loss": Bloss.item(),
                "Lcls_loss": Lloss.item()
            }
            return loss, status
        else:
            return loss
