import os
import os.path as osp
import zipfile
import shutil
import glob
import json
import argparse
import cv2
import numpy as np


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def not_exist(pred, threshold=0.5):
    # print(pred)
    return (len(pred) == 5 and pred[-1] < threshold) or (len(pred) == 1 and pred[0] == 0) or len(pred) == 0


def eval(out_res, label_res):
    measure_per_frame = []
    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
        measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred[:4], _gt[:4]) if len(_pred) > 1 else 0)
    return np.mean(measure_per_frame)


def eval_offline(out_res_dir, label_res_dir, datamode, save_filename=None, mode='IR'):
    with open(osp.join(label_res_dir, 'split.bk.json')) as f:
        data_split = json.load(f)

    # prepare video list
    seq_list = []
    for video_name in (data_split['train'] + data_split['val'] if datamode == 'all' else data_split[datamode]):
        video_name = video_name.split('/')[-1]
        if osp.exists(osp.join(out_res_dir, video_name + '.txt')):
            seq_list.append(video_name)

            # evaluation
    # import pdb
    # pdb.set_trace()
    overall_performance = []
    for video_id, video_name in enumerate(seq_list):
        with open(osp.join(out_res_dir, video_name + '.txt')) as f:
            out_res = np.loadtxt(f, dtype=float).tolist()
        with open(osp.join(out_res_dir, video_name + '_conf.txt')) as f:
            out_res_conf = [1.0] + list(np.loadtxt(f, dtype=float))
        for i in range(len(out_res)):
            out_res[i].append(out_res_conf[i])
        with open(osp.join(label_res_dir, 'trainval', video_name, mode + '_label.json')) as f:
            label_res = json.load(f)
        mixed_measure = eval(out_res, label_res)
        overall_performance.append(mixed_measure)
        print('[%03d/%03d] %20s %5s Fixed Measure: %.04f' % (video_id, len(seq_list), video_name, mode, mixed_measure))
    print('[Overall] %5s Mixed Measure: %.04f\n' % (mode, np.mean(overall_performance)))

    # save according to the competition standard
    if save_filename is not None:
        if osp.exists('./submission/'):
            shutil.rmtree('./submission/')
        os.makedirs('./submission/tmp')

        for video_name in seq_list:
            with open(osp.join(out_res_dir, video_name + '.txt')) as f:
                out_res = np.loadtxt(f, dtype=float).tolist()
            with open(osp.join(out_res_dir, video_name + '_conf.txt')) as f:
                out_res_conf = [1.0] + list(np.loadtxt(f, dtype=float))
            for i in range(len(out_res)):
                out_res[i].append(out_res_conf[i])
            with open(osp.join(label_res_dir, 'trainval', video_name, mode + '_label.json')) as f:
                label_res = json.load(f)
            with open(osp.join('./submission/tmp', video_name.split('/')[-1] + '_%s.txt' % mode), 'w') as f:
                res = []
                for i, frame_res in enumerate(out_res):
                    if i == 0:
                        res.append(label_res['gt_rect'][0])
                        continue
                    if not_exist(frame_res):
                        res.append([0])
                    else:
                        res.append(frame_res[:4])
                json.dump({'res': res}, f)
        make_zip('./submission/tmp/', save_filename)
        shutil.rmtree('./submission/')


def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)  # 相对路径
            zipf.write(pathfile, arcname)
    zipf.close()


def main(mode='IR', visulization=False):
    assert mode in ['IR', 'RGB'], 'Only Support IR or RGB to evalute'
    # setup tracker
    net_path = 'model.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    # setup experiments
    video_paths = glob.glob(os.path.join('dataset', 'test-challenge', '*'))
    video_num = len(video_paths)
    output_dir = os.path.join('results', tracker.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    overall_performance = []

    # run tracking experiments and report performance
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        video_file = os.path.join(video_path, '%s.mp4' % mode)
        res_file = os.path.join(video_path, '%s_label.json' % mode)
        with open(res_file, 'r') as f:
            label_res = json.load(f)

        init_rect = label_res['gt_rect'][0]
        capture = cv2.VideoCapture(video_file)

        frame_id = 0
        out_res = []
        while True:
            ret, frame = capture.read()
            if not ret:
                capture.release()
                break
            if frame_id == 0:
                tracker.init(frame, init_rect)  # initialization
                out = init_rect
                out_res.append(init_rect)
            else:
                out = tracker.update(frame)  # tracking
                out_res.append(out.tolist())
            if visulization:
                _gt = label_res['gt_rect'][frame_id]
                _exist = label_res['exist'][frame_id]
                if _exist:
                    cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                                  (0, 255, 0))
                cv2.putText(frame, 'exist' if _exist else 'not exist',
                            (frame.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0) if _exist else (0, 0, 255), 2)

                cv2.rectangle(frame, (int(out[0]), int(out[1])), (int(out[0] + out[2]), int(out[1] + out[3])),
                              (0, 255, 255))
                cv2.imshow(video_name, frame)
                cv2.waitKey(1)
            frame_id += 1
        if visulization:
            cv2.destroyAllWindows()
        # save result
        output_file = os.path.join(output_dir, '%s_%s.txt' % (video_name, mode))
        with open(output_file, 'w') as f:
            json.dump({'res': out_res}, f)

        mixed_measure = eval(out_res, label_res)
        overall_performance.append(mixed_measure)
        print('[%03d/%03d] %20s %5s Fixed Measure: %.03f' % (video_id, video_num, video_name, mode, mixed_measure))

    print('[Overall] %5s Mixed Measure: %.03f\n' % (mode, np.mean(overall_performance)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datamode', choices=['train', 'val', 'all'], default='all')
    parser.add_argument('--save_filename', default='Super_DiMP.zip')
    args = parser.parse_args()
    # main(mode='IR', visulization=False)
    eval_offline('/mnt/data3/tzh/Stark/test/tracking_results/stark_st/baseline_anti_only/',
                 '/home/dsz/datasets/anti-uav', datamode=args.datamode, save_filename=args.save_filename)
