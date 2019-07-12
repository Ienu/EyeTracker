# -*- coding: utf-8 -*-
'''
FileName: detect_face.py
Author:   Wenyu
Date:     07/11/2019
Version:  v1.0 [07/11/2019][Wenyu] detect face in the image by Yolo v3
'''

from .tiny_yolo.yolo import YOLO

import cv2
import numpy as np
import argparse

from PIL import Image


class FaceDetect:

    def __init__(self):
        '''init function'''

        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        '''Command line options'''
        parser.add_argument('--model', type=str, help='path to model weight file, default ' + YOLO.get_defaults("model_path"))
        parser.add_argument('--anchors', type=str, help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path"))
        parser.add_argument('--classes', type=str, help='path to class definitions, default ' + YOLO.get_defaults("classes_path"))
        parser.add_argument('--gpu_num', type=int, help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num")))
        parser.add_argument('--image', default=False, action="store_true", help='Image detection mode, will ignore all positional arguments')
        '''Command line positional arguments -- for video detection mode'''
        parser.add_argument("--input", nargs='?', type=str,required=False,default='./path2your_video', help = "Video input path")
        parser.add_argument("--output", nargs='?', type=str, default="", help="[Optional] Video output path")

        FLAGS = parser.parse_args()
        
        self.yolo = YOLO(**vars(FLAGS))

    def detect_face(self, image):
        '''get face bounding box from image'''

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(image)
        face = self.yolo.get_box(img)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if face == None:
            return

        if len(face) > 0:
            return face

