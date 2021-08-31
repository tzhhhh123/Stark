from . import STARKSActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.box_ops import box_iou
import torch


class STARKSTActor(STARKSActor):
    """ Actor for training the STARK-ST(Stage2)"""

    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective, loss_weight, settings)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.loss_type = 'BCE'

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
        out_dict = self.forward_pass(data, run_box_head=False, run_cls_head=True)

        # process the groundtruth label
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        loss, status = self.compute_losses(out_dict, gt_bboxes[0])

        return loss, status

    def compute_losses(self, pred_dict, gt_bbox, return_status=True, only_cls=True):
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        nlp_pred_boxes_vec = box_cxcywh_to_xyxy(pred_dict['nlp_pred_boxes']).view(-1, 4)
        iou, nlp_iou = box_iou(pred_boxes_vec, gt_boxes_vec)[0], box_iou(nlp_pred_boxes_vec, gt_boxes_vec)[0],
        if self.loss_type == 'BCE':
            cls_loss = torch.nn.BCELoss()
            iou_arg = (iou < 0.3) | (iou > 0.7)
            nlp_arg = (nlp_iou < 0.3) | (nlp_iou > 0.7)
            iou[iou < 0.3] = 0
            iou[iou > 0.7] = 1
            nlp_iou[nlp_iou < 0.3] = 0
            nlp_iou[nlp_iou > 0.7] = 1
            Bloss = cls_loss(pred_dict["pred_logits"].view(-1)[iou_arg], iou[iou_arg])
            Lloss = cls_loss(pred_dict["nlp_pred_logits"].view(-1)[nlp_arg], nlp_iou[nlp_arg])
        else:
            cls_loss = torch.nn.SmoothL1Loss()
            Bloss = cls_loss(pred_dict["pred_logits"].view(-1), iou)
            Lloss = cls_loss(pred_dict["nlp_pred_logits"].view(-1), nlp_iou)

        import random
        if random.random() > 0.995:
            print("pred    ", pred_dict["nlp_pred_logits"].view(-1) - pred_dict["pred_logits"].view(-1))
            print("iou     ", nlp_iou - iou)
            # print("nlp_pred", pred_dict["nlp_pred_logits"].view(-1))
            # print("nlp_iou ", nlp_iou)

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
