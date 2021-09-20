from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.box_ops import box_iou
import torch


class STARKSActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""

    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

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
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=False)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        # compute losses
        loss, status = self.compute_losses(out_dict, gt_bboxes[0])

        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head, only=None):
        feat_dict_list = []
        # process the templates
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            feat_dict_list.append(self.net(img=NestedTensor(template_img_i, template_att_i), mode='backbone'))

        # process the search regions (t-th frame)
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        feat_dict_list.append(self.net(img=NestedTensor(search_img, search_att), mode='backbone'))

        # run the transformer and compute losses

        caption = (data['words_hidden'], data['words_masks'].transpose(0, 1),
                   data['words_pool']) if self.settings.caption else None
        out_dict, _, _ = self.net(seq_dict=feat_dict_list, mode="transformer", run_box_head=run_box_head,
                                  run_cls_head=run_cls_head, caption=caption, only=only)  ###tzh add cap
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def compute_losses(self, pred_dict, gt_bbox, return_status=True):
        # Get boxes
        # num_queries = pred_boxes.size(1)
        num_queries = 1
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        if "pred_boxes" in pred_dict:
            pred_boxes = pred_dict['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            # weighted sum
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        # loss_all = loss
        if 'nlp_pred_boxes' in pred_dict:
            nlp_pred_boxes = pred_dict['nlp_pred_boxes']
            nlp_pred_boxes_vec = box_cxcywh_to_xyxy(nlp_pred_boxes).view(-1, 4)
            try:
                nlp_giou_loss, nlp_iou = self.objective['giou'](nlp_pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                nlp_giou_loss, nlp_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

            nlp_l1_loss = self.objective['l1'](nlp_pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            nlp_loss = self.loss_weight['giou'] * nlp_giou_loss + self.loss_weight['l1'] * nlp_l1_loss

            # loss_all = loss_all + nlp_loss

        if 'fuse_pred_boxes' in pred_dict:
            fuse_pred_boxes = pred_dict['fuse_pred_boxes']
            fuse_pred_boxes_vec = box_cxcywh_to_xyxy(fuse_pred_boxes).view(-1, 4)
            try:
                fuse_giou_loss, fuse_iou = self.objective['giou'](fuse_pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                fuse_giou_loss, fuse_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

            fuse_l1_loss = self.objective['l1'](fuse_pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            fuse_loss = self.loss_weight['giou'] * fuse_giou_loss + self.loss_weight['l1'] * fuse_l1_loss

            # loss_all = loss_all + fuse_loss * 2

        loss_all = 0.5 * loss + 0.5 * fuse_loss + 0.2 * nlp_loss
        # loss_all = fuse_loss
        if return_status:
            # status for log
            status = {"Loss": loss_all.item()}
            if 'nlp_pred_boxes' in pred_dict:
                mean_iou = iou.detach().mean()
                status["B/Loss"] = loss.item()
                status["B/IOU"] = mean_iou.item()
            if 'nlp_pred_boxes' in pred_dict:
                nlp_mean_iou = nlp_iou.detach().mean()
                status["L/Loss"] = nlp_loss.item()
                status["L/IOU"] = nlp_mean_iou.item()
            if 'fuse_pred_boxes' in pred_dict:
                fuse_mean_iou = fuse_iou.detach().mean()
                status["F/Loss"] = fuse_loss.item()
                status["F/IOU"] = fuse_mean_iou.item()
            return loss_all, status
        else:
            return loss_all
