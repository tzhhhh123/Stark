from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
from copy import deepcopy
# for debug
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.stark import build_starkst
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box


class STARK_ST(BaseTracker):
    def __init__(self, params, dataset_name):
        super(STARK_ST, self).__init__(params)
        network = build_starkst(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # template update
        self.z_dict1 = {}
        self.caption = None
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = []
        ###to fix
        # self.init_mode = 0  ##box use template grounding use grounding
        # self.select = 'fuse'  ##target , nlp ,fuse
        self.init_mode = self.cfg.TEST.INIT_MODE
        self.select = self.cfg.TEST.SAVE_MODE  ##target , nlp ,fuse
        self.only = 'nlp'  ##only run single head

        print('use language', self.cfg['TRAIN']['CAPTION'])
        print('init_mode:', self.init_mode)
        print('save_mode:', self.select)
        print("interval:", self.update_intervals)

    def initialize(self, image, info: dict):
        # forward the template once
        H, W, _ = image.shape
        ##add captioon
        self.caption = (
            torch.from_numpy(info['last_hidden_state']).cuda().unsqueeze(1),
            torch.from_numpy(info['masks']).cuda().unsqueeze(0),
            torch.from_numpy(info['pool_out']).cuda().unsqueeze(1)) \
            if self.cfg['TRAIN']['CAPTION'] else None

        if self.init_mode == "grounding":
            ###make a fake template
            self.state = [0, 0, W, H]
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state, 1,
                                                        output_sz=self.params.search_size)  # (x1, y1, w, h)
            template = self.preprocessor.process(z_patch_arr, z_amask_arr)
            with torch.no_grad():
                self.z_dict1 = self.network.forward_backbone(template)

            ###track a frame
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, 1,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
            search = self.preprocessor.process(x_patch_arr, x_amask_arr)

            with torch.no_grad():
                x_dict = self.network.forward_backbone(search)

                feat_dict_list = [self.z_dict1, x_dict]

                out_dict, _, _ = self.network.forward_transformer(seq_dict=feat_dict_list, run_box_head=True,
                                                                  run_cls_head=False, caption=self.caption,
                                                                  only=None)
            nlp_pred_boxes = out_dict['nlp_pred_boxes'].view(-1, 4)

            nlp_pred_box = (nlp_pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

            self.state = clip_box(self.map_box_back(nlp_pred_box, resize_factor), H, W, margin=10)

        else:
            self.state = info['init_bbox']
        z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)

        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template)
        # save states
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None, only=None):
        H, W, _ = image.shape
        self.frame_id += 1
        # get the t-th search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = [self.z_dict1, x_dict]
            # seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            # only = 'nlp' if self.test_mode == 1 else None
            out_dict, _, _ = self.network.forward_transformer(seq_dict=feat_dict_list, run_box_head=True,
                                                              run_cls_head=False, caption=self.caption, only=self.only)
        # import ipdb
        # ipdb.set_trace()
        pred = {}
        if 'pred_boxes' in out_dict:
            pred_boxes = out_dict['pred_boxes'].view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            pred['target_bbox'] = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if 'fuse_pred_boxes' in out_dict:
            fuse_pred_boxes = out_dict['fuse_pred_boxes'].view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            fuse_pred_box = (fuse_pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            pred['fuse_bbox'] = clip_box(self.map_box_back(fuse_pred_box, resize_factor), H, W, margin=10)

        if 'nlp_pred_boxes' in out_dict:
            nlp_pred_boxes = out_dict['nlp_pred_boxes'].view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            nlp_pred_box = (nlp_pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            pred['nlp_bbox'] = clip_box(self.map_box_back(nlp_pred_box, resize_factor), H, W, margin=10)

        self.state = pred[self.select + '_bbox']

        if "pred_logits" in out_dict:
            pred['pred_logits'] = out_dict["pred_logits"]
        if "pred_logits" in out_dict:
            pred['nlp_pred_logits'] = out_dict["nlp_pred_logits"]
        # update template
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    self.z_dict1 = self.network.forward_backbone(template_t)

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        return pred
        # if self.save_all_boxes:###not use
        #     '''save all predictions'''
        #     all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
        #     all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
        #     return {"target_bbox": self.state,
        #             "all_boxes": all_boxes_save}
        # else:

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return STARK_ST
