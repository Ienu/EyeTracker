# -*- coding: utf-8 -*-
'''
FileName: detect_face.py
Author:   Wenyu
Date:     07/12/2019
Version:  v1.0 [07/12/2019][Wenyu] detect landmarks in face image by SBR
          v1.1 [07/12/2019][Wenyu] use CPU only as torch needs CUDA > 3.0
'''

import cv2
import numpy as np

import torch

from .SBR.lib.xvision import transforms, draw_image_by_points
from .SBR.lib.models import obtain_model, remove_module_dict
from .SBR.lib.config_utils import load_configure
from .SBR.lib.datasets import GeneralDataset as Dataset


class LandmarkDetect:

    def __init__(self):
        '''init function'''
        model_path = 'E:/Collection/Study/EyeTracker/landmark_detect/SBR/snapshots/CPM-SBR/checkpoint/cpm_vgg16-epoch-049-050.pth'
        config_path = 'E:/Collection/Study/EyeTracker/landmark_detect/SBR/configs/Detector.config'
        #assert torch.cuda.is_available(), 'CUDA is not available.'
        torch.backends.cudnn.enabled = False#True
        torch.backends.cudnn.benchmark = False#True
        snapshot = torch.load(model_path)

        # General Data Argumentation
        #mean_fill = tuple([int(x * 255) for x in [0.485, 0.456, 0.406]])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.param = snapshot['args']
        # import pdb; pdb.set_trace()
        eval_transform = transforms.Compose([transforms.PreCrop(self.param.pre_crop_expand), transforms.TrainScale2WH((self.param.crop_width, self.param.crop_height)), transforms.ToTensor(), normalize])
        model_config = load_configure(config_path, None)
        self.dataset = Dataset(eval_transform, self.param.sigma, model_config.downsample, self.param.heatmap_type, self.param.data_indicator)
        self.dataset.reset(self.param.num_pts)

        self.net = obtain_model(model_config, self.param.num_pts + 1)
        #self.net = self.net.cuda()
        weights = remove_module_dict(snapshot['detector'])
        self.net.load_state_dict(weights)

    def detect_landmarks(self, img, face):
        '''detect landmarks'''
        [sbr_image, _, _, _, _, _, cropped_size], meta = self.dataset.prepare_input(img, [face[1], face[2], face[3], face[4]])
        inputs = sbr_image.unsqueeze(0)#.cuda()
        # network forward
        with torch.no_grad():
            batch_heatmaps, batch_locs, batch_scos = self.net(inputs)
        # obtain the locations on the image in the orignial size
        cpu = torch.device('cpu')
        np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(cpu).numpy(), cropped_size.numpy()
        locations, scores = np_batch_locs[0, :-1, :], np.expand_dims(np_batch_scos[0, :-1], -1)

        scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2), cropped_size[1] * 1. / inputs.size(-1)

        locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + cropped_size[3]
        prediction = np.concatenate((locations, scores), axis=1).transpose(1, 0)
        
        return prediction
