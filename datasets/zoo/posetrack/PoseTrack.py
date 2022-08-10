#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import os.path as osp
import torch
import copy
import random
import cv2
from pycocotools.coco import COCO
import logging
from collections import OrderedDict
from tabulate import tabulate
from termcolor import colored

from .posetrack_utils.posetrack_utils import video2filenames #, evaluate_simple
from utils.utils_json import read_json_from_file, write_json_to_file
from utils.utils_bbox import box2cs
from utils.utils_image import read_image
from utils.utils_folder import create_folder
from utils.utils_registry import DATASET_REGISTRY
from datasets.process import get_affine_transform, fliplr_joints, exec_affine_transform, generate_heatmaps, half_body_transform, \
    convert_data_to_annorect_struct

from datasets.transforms import build_transforms
from datasets.zoo.base import VideoDataset

from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE


@DATASET_REGISTRY.register()
class PoseTrack(VideoDataset):
    """
        PoseTrack
    """

    def __init__(self, cfg, phase, **kwargs):
        super(PoseTrack, self).__init__(cfg, phase, **kwargs)

        self.train = True if phase == TRAIN_PHASE else False
        
        # posetrack 
        #self.flip_pairs = [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        #self.joints_weight = np.array([1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5],
        #                              dtype=np.float32).reshape((self.num_joints, 1))
        #self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        #self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        #self.is_posetrack18 = cfg.DATASET.IS_POSETRACK18
        #self.transform = build_transforms(cfg, phase)
        
        # h36m ------------------------------------------- 
        self.flip_pairs = [[1, 4], [2, 5], [3, 6], [11, 14], [12, 15], [13, 16]]
        self.joints_weight = np.array([1., 1., 1.2, 1.5, 1., 1.2, 1.5, 1., 1., 1., 1., 1., 1.2, 1.5, 1., 1.2, 1.5],
                                      dtype=np.float32).reshape((self.num_joints, 1))
        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_ids = (0,1,2,3,4,5,6)

        self.is_posetrack18 = cfg.DATASET.IS_POSETRACK18
        self.transform = build_transforms(cfg, phase)        

        # _C.DISTANCE_WHOLE_OTHERWISE_SEGMENT = True
        #_C.DISTANCE = 2
        #_C.PREVIOUS_DISTANCE = 1
        #_C.NEXT_DISTANCE = 1
        self.distance_whole_otherwise_segment = cfg.DISTANCE_WHOLE_OTHERWISE_SEGMENT
        self.distance = cfg.DISTANCE
        self.previous_distance = cfg.PREVIOUS_DISTANCE
        self.next_distance = cfg.NEXT_DISTANCE

        self.random_aux_frame = cfg.DATASET.RANDOM_AUX_FRAME

        # 1.25 
        self.bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        # sigma = 3
        self.sigma = cfg.MODEL.SIGMA

        # ------------------------------------------------------------config 파일 수정 필요!! 
        # self.img_dir = cfg.DATASET.IMG_DIR
        # self.json_dir = cfg.DATASET.JSON_DIR
        # self.test_on_train = cfg.DATASET.TEST_ON_TRAIN
        # self.json_file = cfg.DATASET.JSON_FILE
        # "/home/jongmin2/H36M-Toolbox/processed/h36m_coco_train_5fps.json"
        self.root_path ="/home/jongmin2/H36M-Toolbox/processed"
        self.data_version = "h36m_coco_train_5fps"
        self.annotation_path = osp.join(self.root_path, self.data_version +'.json')

        
        #val, test 시 필요함!!
        if self.phase != TRAIN_PHASE:
             self.img_dir = cfg.DATASET.TEST_IMG_DIR
             temp_subCfgNode = cfg.VAL if self.phase == VAL_PHASE else cfg.TEST
             self.nms_thre = temp_subCfgNode.NMS_THRE
             self.image_thre = temp_subCfgNode.IMAGE_THRE
             self.soft_nms = temp_subCfgNode.SOFT_NMS
             self.oks_thre = temp_subCfgNode.OKS_THRE
             self.in_vis_thre = temp_subCfgNode.IN_VIS_THRE
             self.bbox_file = temp_subCfgNode.COCO_BBOX_FILE
             self.use_gt_bbox = temp_subCfgNode.USE_GT_BBOX
             self.annotation_dir = temp_subCfgNode.ANNOT_DIR

        #self.coco = COCO(osp.join(self.json_dir, 'posetrack_train.json' if self.is_train else 'posetrack_val.json'))
        self.coco = COCO(self.annotation_path)

        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        
        # 이 부분이 왜 있는지에 대한 study가 좀 더 필요함 --------------------------------------------------------------------
        #self.classes = ['__background__'] + cats
        #self.num_classes = len(self.classes)
        #self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        #self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        #self._coco_ind_to_class_ind = dict(
        #    [(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]])
        
        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)
  
        # data = self._load_coco_keypoints_annotations()      
        self.data = self._list_data()

        self.model_input_type = cfg.DATASET.INPUT_TYPE

        self.show_data_parameters()
        self.show_samples()



    def __getitem__(self, item_index):
        # 기본적으로 하나의 이미지이냐? 비디오인가에 따라서 구분을 한다. !! 
        data_item = copy.deepcopy(self.data[item_index])
        
        # -----------------------------------------------jongmin added code 
        # pre, next input image code!!
        #print(item_index)

        if item_index == 0:
            pre_filename = ""
            pass
        else:
            pre_data_item = copy.deepcopy(self.data[item_index-1])
            pre_filename = pre_data_item['image']
        
        # next를 따질 때... 전체 비디오프레임 수를 넘을 수는 없기 때문에 .. 해당 부분처리를 위해 추가 
        if item_index == (self.num_images-1):
            next_filename = ""
            pass         
        else:
            next_data_item = copy.deepcopy(self.data[item_index+1])
            next_filename = next_data_item['image']      

        # -------------------------------------------------
            
        if self.model_input_type == 'single_frame':
            return self._get_single_frame(data_item)
        elif self.model_input_type == 'spatiotemporal_window':
            return self._get_spatiotemporal_window(data_item,pre_filename,next_filename,item_index)

    # dataset 관련해서 입력이 되는 코드 !! 
    # 해당 부분을 posetrack에서 h36m으로 변경을 하면 됨. 
    def _get_spatiotemporal_window(self, data_item,pre_filename,next_filename,item_index):
        
        # dcpose 기준으로 filename은 비워 있는 값 
        filename = data_item['filename']
        # 'imgnum': 0,
        img_num = data_item['imgnum']
        
        # 이미지 경로를 통해 불러온다라고 생각하면 됨. 
        image_file_path = data_item['image']
        origin_path = data_item['origin_path']
        
        #num_frames = data_item['nframes']
        # read BGR image
        data_numpy = read_image(image_file_path)

        # 이 부분의 의미는 단순하게 .. 파일명의 길이를 보기 위한 코드임!! 
        # zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))

        #if zero_fill == 6:
        #    is_posetrack18 = True
        #else:
        #    is_posetrack18 = False

        # 안 맞는 부분!! 
        # h36m과 posetrack이름 차이로 발생 !! -----------------------------------------수정 필요 
        #current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))

        # _C.DISTANCE_WHOLE_OTHERWISE_SEGMENT = True
        # if self.distance_whole_otherwise_segment:
        #     farthest_distance = self.distance
            
        #     # 만약 이미지 idx 0 이라면, range(1,1)
        #     # 만약 이미지 idx 1 이라면, range(1,2)
        #     # 만약 이미지 idx 2 이라면, range(1,2)
        #     # 만약 이미지 idx 3 이라면, range(1,2)
        #     # 만약 이미지 idx 4 이라면, range(1,2)
        #     prev_delta_range = range(1, min((current_idx + 1) if is_posetrack18 else current_idx, farthest_distance))
        #     next_delta_range = range(1, min((num_frames - current_idx) if is_posetrack18 else (num_frames - current_idx + 1),
        #                                     farthest_distance))
        # else:
        #     prev_delta_range = range(1, min(current_idx + 1 if is_posetrack18 else current_idx, self.previous_distance))
        #     next_delta_range = range(1, min((num_frames - current_idx) if is_posetrack18 else (num_frames - current_idx + 1),
        #                                     self.next_distance))

        # prev_delta_range = list(prev_delta_range)
        # next_delta_range = list(next_delta_range)
        # # setting deltas

        # if len(prev_delta_range) == 0:
        #     prev_delta = 0
        #     margin_left = 1
        # else:
        #     if self.random_aux_frame:
        #         prev_delta = random.choice(prev_delta_range)
        #     else:
        #         prev_delta = prev_delta_range[-1]
        #     margin_left = prev_delta

        # if len(next_delta_range) == 0:
        #     next_delta = 0
        #     margin_right = 1
        # else:
        #     if self.random_aux_frame:
        #         next_delta = random.choice(next_delta_range)
        #     else:
        #         next_delta = next_delta_range[-1]

        #     margin_right = next_delta

        # # 이 부분에서 최종적으로 pre_idx를 정하는구나!! 
        # # 거의 대부분 하나 또는 두개의 차이의 이미지를 봄 
        # # 다음 이미지나 과거 이미지가 없는 경우에는 현재의 이미지를 중복해서 데이터를 넣어서 처리를 함!!
        # prev_idx = current_idx - prev_delta
        # next_idx = current_idx + next_delta

        # posetrack 기준 세팅 새로운 방식이 필요!! 

        #prev_image_file = osp.join(osp.dirname(image_file_path), str(prev_idx).zfill(zero_fill) + '.jpg')
        #next_image_file = osp.join(osp.dirname(image_file_path), str(next_idx).zfill(zero_fill) + '.jpg')
        
        # ----------------------------------------------------------- jongmin add code 
        # 기존 코드와 달리 전 이미지에 대해서만 학습을 한다. 
        # 전 프레임과 다음 프레임 기준으로 학습을 한다. 

        # pre image check code ---------------------------   
        if pre_filename == "":
          pre_image_existed = False 
          optical_image_existed = False
        else:
          pre = pre_filename.split('/')  
          current = image_file_path.split('/')
          
          if pre[-2]==current[-2]:
            pre_image_existed = True
            optical_image_existed = True
          else:
            pre_image_existed = False
            optical_image_existed = False
            
            
        # next image check code ---------------------------            
        if next_filename == "":
          next_image_existed = False 
        else:
          next = next_filename.split('/')  
          current = image_file_path.split('/')
          if next[-2]==current[-2]:
            next_image_existed = True
          else:
            next_image_existed = False            
        
        # 과거와 미래 이미지가 없다면, 현재 프레임을 중복으로 넣어서 계산을 한다. 
        if pre_image_existed:
            prev_image_file = pre_filename
        else:
            prev_image_file = image_file_path        
            

        if optical_image_existed:
            optical_path = osp.join(self.root_path,"images_optical", origin_path)
            optical_image = read_image(optical_path)
            print("optical image -> read image")
        else:
            optical_image = None
            print("optical image -> none")
                    
            
        if next_image_existed:     
            next_image_file = next_filename
        else:
            next_image_file = image_file_path

        # checking for files existence
        # if not osp.exists(prev_image_file):
        #     error_msg = "Can not find image :{}".format(prev_image_file)
        #     self.logger.error(error_msg)
        #     raise Exception(error_msg)
        # if not osp.exists(next_image_file):
        #     error_msg = "Can not find image :{}".format(next_image_file)
        #     self.logger.error(error_msg)
        #     raise Exception(error_msg)

        #read_image(image_file_path)
        data_numpy_prev = read_image(prev_image_file)
        data_numpy_next = read_image(next_image_file)
        
        #data_numpy_prev = read_image(image_file_path)
        #data_numpy_next = read_image(image_file_path)

        if self.color_rgb:
            # cv2 read_image  color channel is BGR
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            data_numpy_prev = cv2.cvtColor(data_numpy_prev, cv2.COLOR_BGR2RGB)
            data_numpy_next = cv2.cvtColor(data_numpy_next, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            self.logger.error('=> fail to read {}'.format(image_file_path))
            raise ValueError('Fail to read {}'.format(image_file_path))
        if data_numpy_prev is None:
            self.logger.error('=> PREV SUP: fail to read {}'.format(prev_image_file))
            raise ValueError('PREV SUP: Fail to read {}'.format(prev_image_file))
        if data_numpy_next is None:
            self.logger.error('=> NEXT SUP: fail to read {}'.format(next_image_file))
            raise ValueError('NEXT SUP: Fail to read {}'.format(next_image_file))

        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']

        center = data_item["center"]
        scale = data_item["scale"]

        score = data_item['score'] if 'score' in data_item else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints, self.upper_body_ids, self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            # scale = scale * np.random.uniform(1 - scale_factor[0], 1 + scale_factor[1])
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                data_numpy_prev = data_numpy_prev[:, ::-1, :]
                data_numpy_next = data_numpy_next[:, ::-1, :]
                
                if optical_image is None:
                    pass
                else:
                    optical_image = optical_image[:, ::-1, :]
                    
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        # calculate transform matrix
        trans = get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_prev = cv2.warpAffine(data_numpy_prev, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_next = cv2.warpAffine(data_numpy_next, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        
        current_image_numpy = input_x.copy()
        
        if optical_image is None:
            pass
        else:
            optical_image = cv2.warpAffine(optical_image, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)
            input_prev = self.transform(input_prev)
            input_next = self.transform(input_next)

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]
        # target_heatmaps, target_heatmaps_weight = self._generate_target(joints, joints_vis, self.heatmap_size, self.num_joints)

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma, self.image_size, self.heatmap_size,
                                                                    self.num_joints, optical_image,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
                                                                    
                                                                    
        #output = self.output_heatmaps(current_image_numpy,target_heatmaps)
        #file_name = 'check/{}.jpg'.format(item_index)
        #cv2.imwrite(file_name,output)                                                                    
                                                                    
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)

        meta = {
            'image': image_file_path,                # 현재 이미지 경로
            'prev_sup_image': prev_image_file,       # 과거 이미지 경로
            'next_sup_image': next_image_file,       # 미래 이미지 경로
            'filename': filename, # 의미 없음.
            'imgnum': img_num, # 의미 없음.
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
            #'margin_left': margin_left,
            #'margin_right': margin_right,
            # 임시
            'margin_left': 1,
            'margin_right': 1,
        }

        return input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, meta

    # cv2 이미지어야함. - tensor가 아니라 
    # heatmaps 또한 tensor형태가 아니여햐함.
    def output_heatmaps(self, image, heatmaps):
    
        heatmaps = heatmaps*255
        heatmaps = heatmaps.astype('uint8')
        
        num_joints, height, width = heatmaps.shape
        image_resized = cv2.resize(image, (int(width), int(height)))

        image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)
    
        for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
            heatmap = heatmaps[j, :, :]
            
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            image_fused = colored_heatmap*0.7 + image_resized*0.3
    
            width_begin = width * (j+1)
            width_end = width * (j+2)
            image_grid[:, width_begin:width_end, :] = image_fused
    
        image_grid[:, 0:width, :] = image_resized
    
        return image_grid


    def _get_single_frame(self, data_item):
        raise NotImplementedError

    def _list_data(self):
        if self.is_train or self.use_gt_bbox:
            # use bbox from annotation
            data = self._load_coco_keypoints_annotations()
        else:
            # use bbox from detection
            data = self._load_detection_results()
        return data

    def _load_coco_keypoints_annotations(self):
        """
            coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
            iscrowd:
                crowd instances are handled by marking their overlaps with all categories to -1
                and later excluded in training
            bbox:
                [x1, y1, w, h]
        """
        gt_db = []
        for index in self.image_set_index:
            im_ann = self.coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']

            file_name = im_ann['file_name']

            # 해당 부분에 대한 데이터가 없거나, 유사한 데이터를 찾아야함 !! ------------------------------------------------
            #nframes = int(im_ann['nframes'])
            #frame_id = int(im_ann['frame_id'])

            annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
            objs = self.coco.loadAnns(annIds)

            # sanitize bboxes
            valid_objs = []
            for obj in objs:
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    valid_objs.append(obj)
            objs = valid_objs

            rec = []
            for obj in objs:
                
                # h36m은 무조건 cls 1
                #cls = self._coco_ind_to_class_ind[obj['category_id']]
                #if cls != 1:
                #    continue

                # ignore objs without keypoints annotation
                if max(obj['keypoints']) == 0:
                    continue

                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
                for ipt in range(self.num_joints):
                    joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints_3d[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                # self.bbox_enlarge_factor = 1.25로 구성 !!
                center, scale = box2cs(obj['clean_bbox'][:4], self.aspect_ratio, self.bbox_enlarge_factor)
                rec.append({
                    # posetrack18 기준 json 
                    # "file_name": "images/val/009478_mpii_test/000000.jpg"
                    'image': osp.join(self.root_path,"images", self.coco.loadImgs(index)[0]['file_name']),
                    'origin_path': self.coco.loadImgs(index)[0]['file_name'],
                    'center': center,
                    'scale': scale,
                    'box': obj['clean_bbox'][:4],
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    #'nframes': nframes,
                    #'frame_id': frame_id,
                })
            gt_db.extend(rec)
        return gt_db

    def _load_detection_results(self):
        logger = logging.getLogger(__name__)
        logger.info("=> Load bbox file from {}".format(self.bbox_file))
        all_boxes = read_json_from_file(self.bbox_file)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        kpt_data = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = det_res['image_name']
            box = det_res['bbox']  # xywh
            score = det_res['score']
            
            # frame_id는 여기서 왜 필요한 것인가??? 
            # nframes는 h36m json에서는 별도로 없음. 새로 만들거나 현재는 사용이 불가능 
            # nframes의 목적은 각 비디오 당 전체 프레임 수를 알기 위한 목적이다.
            nframes = det_res['nframes']
            frame_id = det_res['frame_id']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = box2cs(box, self.aspect_ratio, self.bbox_enlarge_factor)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_data.append({
                'image': osp.join(self.img_dir, img_name),
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                #'nframes': nframes,
                #'frame_id': frame_id,
            })
        # logger.info('=> Total boxes: {}'.format(len(all_boxes)))
        # logger.info('=> Total boxes after filter low score@{}: {}'.format(self.image_thre, num_boxes))

        table_header = ["Total boxes", "Filter threshold", "Remaining boxes"]
        table_data = [[len(all_boxes), self.image_thre, num_boxes]]
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Boxes Info Table: \n" + colored(table, "magenta"))

        return kpt_data

    # def evaluate(self, cfg, preds, output_dir, boxes, img_path, *args, **kwargs):
    #     logger = logging.getLogger(__name__)
    #     logger.info("=> Start evaluate")
    #     if self.phase == 'validate':
    #         output_dir = osp.join(output_dir, 'val_set_json_results')
    #     else:
    #         output_dir = osp.join(output_dir, 'test_set_json_results')

    #     create_folder(output_dir)

    #     ### processing our preds
    #     video_map = {}
    #     vid2frame_map = {}
    #     vid2name_map = {}

    #     all_preds = []
    #     all_boxes = []
    #     all_tracks = []
    #     cc = 0

    #     # print(img_path)
    #     for key in img_path:
    #         temp = key.split('/')

    #         # video_name = osp.dirname(key)
    #         video_name = temp[len(temp) - 3] + '/' + temp[len(temp) - 2]
    #         img_sfx = temp[len(temp) - 3] + '/' + temp[len(temp) - 2] + '/' + temp[len(temp) - 1]

    #         prev_nm = temp[len(temp) - 1]
    #         frame_num = int(prev_nm.replace('.jpg', ''))
    #         if not video_name in video_map:
    #             video_map[video_name] = [cc]
    #             vid2frame_map[video_name] = [frame_num]
    #             vid2name_map[video_name] = [img_sfx]
    #         else:
    #             video_map[video_name].append(cc)
    #             vid2frame_map[video_name].append(frame_num)
    #             vid2name_map[video_name].append(img_sfx)

    #         idx_list = img_path[key]
    #         pose_list = []
    #         box_list = []
    #         for idx in idx_list:
    #             temp = np.zeros((4, 17))
    #             temp[0, :] = preds[idx, :, 0]
    #             temp[1, :] = preds[idx, :, 1]
    #             temp[2, :] = preds[idx, :, 2]
    #             temp[3, :] = preds[idx, :, 2]
    #             pose_list.append(temp)

    #             temp = np.zeros((1, 6))
    #             temp[0, :] = boxes[idx, :]
    #             box_list.append(temp)

    #         all_preds.append(pose_list)
    #         all_boxes.append(box_list)
    #         cc += 1

    #     annot_dir = self.annotation_dir
    #     is_posetrack18 = self.is_posetrack18

    #     out_data = {}
    #     out_filenames, L = video2filenames(annot_dir)

    #     for vid in video_map:
    #         idx_list = video_map[vid]
    #         c = 0
    #         used_frame_list = []
    #         cur_length = L['images/' + vid]

    #         temp_kps_map = {}
    #         temp_track_kps_map = {}
    #         temp_box_map = {}

    #         for idx in idx_list:
    #             frame_num = vid2frame_map[vid][c]
    #             img_sfx = vid2name_map[vid][c]
    #             c += 1

    #             used_frame_list.append(frame_num)

    #             kps = all_preds[idx]
    #             temp_kps_map[frame_num] = (img_sfx, kps)

    #             bb = all_boxes[idx]
    #             temp_box_map[frame_num] = bb
    #         #### including empty frames
    #         nnz_counter = 0
    #         next_track_id = 0

    #         if not is_posetrack18:
    #             sid = 1
    #             fid = cur_length + 1
    #         else:
    #             sid = 0
    #             fid = cur_length
    #         # start id ~ finish id
    #         for current_frame_id in range(sid, fid):
    #             frame_num = current_frame_id
    #             if not current_frame_id in used_frame_list:
    #                 temp_sfx = vid2name_map[vid][0]
    #                 arr = temp_sfx.split('/')
    #                 if not is_posetrack18:
    #                     img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(8) + '.jpg'
    #                 else:
    #                     img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(6) + '.jpg'
    #                 kps = []
    #                 tracks = []
    #                 bboxs = []

    #             else:

    #                 img_sfx = temp_kps_map[frame_num][0]
    #                 kps = temp_kps_map[frame_num][1]
    #                 bboxs = temp_box_map[frame_num]
    #                 tracks = [track_id for track_id in range(len(kps))]
    #                 # tracks = [1] * len(kps)

    #             ### creating a data element
    #             data_el = {
    #                 'image': {'name': img_sfx},
    #                 'imgnum': [frame_num],
    #                 'annorect': convert_data_to_annorect_struct(kps, tracks, bboxs),
    #             }
    #             if vid in out_data:
    #                 out_data[vid].append(data_el)
    #             else:
    #                 out_data[vid] = [data_el]

    #     logger.info("=> saving files for evaluation")
    #     #### saving files for evaluation
    #     for vname in out_data:
    #         vdata = out_data[vname]
    #         outfpath = osp.join(output_dir, out_filenames[osp.join('images', vname)])

    #         write_json_to_file({'annolist': vdata}, outfpath)

    #     # run evaluation
    #     # AP = self._run_eval(annot_dir, output_dir)[0]
    #     AP = evaluate_simple.evaluate(annot_dir, output_dir, eval_track=False)[0]

    #     name_value = [
    #         ('Head', AP[0]),
    #         ('Shoulder', AP[1]),
    #         ('Elbow', AP[2]),
    #         ('Wrist', AP[3]),
    #         ('Hip', AP[4]),
    #         ('Knee', AP[5]),
    #         ('Ankle', AP[6]),
    #         ('Mean', AP[7])
    #     ]

    #     name_value = OrderedDict(name_value)

    #     return name_value, name_value['Mean']
