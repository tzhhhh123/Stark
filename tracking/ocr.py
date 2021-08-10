import easyocr
import glob
import json
import tqdm
from multiprocessing import Pool
import cv2
import os, ipdb
import numpy as np

a = json.load(open('/home/dsz/datasets/anti-uav/split.bk.json'))
# a = a['train'] + a['val']
a = a['test']
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


# ipdb.set_trace()


def ocr_save(video_paths):
    reader = easyocr.Reader(['ch_sim'])  # need to run only once to load model into memory
    for b in video_paths:
        print(b)
        dc = {}
        imgs = sorted(glob.glob('/home/dsz/datasets/anti-uav/frames/{}/IR/*.jpg'.format(b)))
        dir = os.path.join('/mnt/data1/tzh/Stark/block_frames_test/{}/IR'.format(b))
        if os.path.exists(dir) is False:
            os.makedirs(dir)

        for fid, img_path in enumerate(imgs):
            # print(b)
            # img = cv2.imread(img_path)
            results = reader.readtext(img_path)
            num = img_path.split('/')[-1]
            dc[num] = results
            # print(result)
        print('down', b)
        json.dump(dc, open('/mnt/data1/tzh/Stark/ocr_res_test/{}.json'.format(b), 'w'))


def iou2d(box1, box2):
    '''
        box [x1,y1,x2,y2]
    '''
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_sum = area1 + area2

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0
    else:
        inter_area = (x2 - x1) * (y2 - y1)
    iou = inter_area / (area_sum - inter_area)
    return iou


def ocr(video_paths):
    reader = easyocr.Reader(['ch_sim'])  # need to run only once to load model into memory
    for b in video_paths:
        print(b)
        imgs = sorted(glob.glob('/home/dsz/datasets/anti-uav/frames/{}/IR/*.jpg'.format(b)))
        dir = os.path.join('/mnt/data1/tzh/Stark/block_frames/{}/IR'.format(b))
        box_st = json.load(open('/home/dsz/datasets/anti-uav/{}/IR_label.json'.format(b)))['gt_rect'][0]
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        results = reader.readtext(imgs[0])
        # print(b)
        # ipdb.set_trace()
        ok = 1
        box_st[2] += box_st[0]
        box_st[3] += box_st[1]
        for res in results:
            boxes = np.array(res[0])
            minx, miny = min(boxes[:, 0]), min(boxes[:, 1])
            maxx, maxy = max(boxes[:, 0]), max(boxes[:, 1])
            if iou2d([minx, miny, maxx, maxy], box_st) > 0:
                ok = 0
        # if b == 'trainval/new35_train-new':
        #     ipdb.set_trace()
        if ok == 0:
            print('error', b)
            continue

        continue
        for img_path in imgs:
            # print(b)
            img = cv2.imread(img_path)
            for res in results:
                boxes = np.array(res[0])
                minx, miny = min(boxes[:, 0]), min(boxes[:, 1])
                maxx, maxy = max(boxes[:, 0]), max(boxes[:, 1])
                img[miny:maxy, minx:maxx, :] = 0
            cv2.imwrite(
                os.path.join('{}/{}'.format(dir, os.path.basename(img_path))), img)

        #     num = img_path.split('/')[-1]
        #     for k in result:
        #         if k.startswith('方位'):
        #             dc[num] = k[2:]
        #     # print(result)
        # print('down', b)
        # json.dump(dc, open('/mnt/data1/tzh/Stark/ocr_res/{}.json'.format(b), 'w'))


# ocr(['20190926_183400_1_8'])
video_paths = []
for b in a:
    # b = b.split('/')[1]
    # if b.startswith('new') is False:
    #     continue
    # ocr(b)
    video_paths.append(b)
# ocr(video_paths)
num_process = 4
p = Pool(num_process)
part_list_len = len(video_paths) // num_process
for i in range(num_process):
    if i == num_process - 1:
        part_list = video_paths[part_list_len * i:]
    else:
        part_list = video_paths[part_list_len * i:part_list_len * (i + 1)]
    p.apply_async(ocr, args=(part_list,))
    print(part_list)
print('Waiting for all subprocesses done...')
p.close()
p.join()
print('All subprocesses done.')
# for k in a:
#     if k.split('/')[1].startswith('new'):
#         print(k)
