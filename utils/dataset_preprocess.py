# -*- coding: utf-8 -*-
'''
FileName: dataset_preprocess.py
Author: Wenyu
Date: 07/26/2019
Version: v1.0 [07/26/2019][Wenyu] obtain whole image from MPIIFaceGaze
'''

import os
import cv2
from scipy.io import loadmat

from utils.log import log


def load_mat(path):
    m = loadmat(path)
    print ("Screen height: {}".format(m['height_pixel']))
    print ("Screen width: {}".format(m['width_pixel']))
    return m['height_pixel'], m['width_pixel']


@log
def dataset_preprocess(dataset_folder):
	'''
	deal with MPIIFaceGaze dataset
	'''
	print('data preprocess...')
	for person in os.listdir(dataset_folder):
		# Determine if it is a folder
		if (os.path.isfile(os.path.abspath(os.path.join(dataset_folder, person))) ):
			continue

		path_to_mat_file = os.path.abspath(os.path.join(dataset_folder, person+'/Calibration/screenSize.mat'))

		# The width and height values are recorded in the calibration data .mat
		height, width = load_mat(path_to_mat_file)

		# Change current dir to person dir
		os.chdir(os.path.abspath(os.path.join(dataset_folder, person)))
		file_name = person + '.txt'
		print (file_name)

		with open(file_name, 'r') as file:
			for line in file.readlines():
				vector = line.split(' ')
				print(vector[0], vector[1], vector[2])

				image_src = cv2.imread(vector[0])

				yield image_src, vector[1], vector[2]

                #face_roi = generate_face_roi(vector[3:15], image_src)
                # cv2.imshow('test', face_roi)
                # cv2.waitKey(1)
                # TODO: according to YOLO v1, HSV color space need to be tested
                # image_dst = cv2.resize(image_src, (face_size, face_size))
                #face_roi = cv2.resize(face_roi, (face_size, face_size))
                # face_data[index, :, :, :] = image_dst
                #face_roi_data[index, :, :, :] = face_roi
                # TODO: according to YOLO v2, they used logistic activation to normalize
                # maybe [0, 1] is better than [-0.5, 0.5] according to Relu
                #eye_track_data[index, :, :, :] = generate_gaze_tensor(vector[1:3], width, height, scale)

		break
