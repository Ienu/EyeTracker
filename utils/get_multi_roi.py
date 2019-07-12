# -*- coding: utf-8 -*-
'''
FileName: get_multi_roi.py
Author: Wenyu
Date: 07/04/2019
Version: v1.0 [07/04/2019][Wenyu] obtain face, face mask, left and right eyes
                                  roi
'''

import cv2
import numpy as np

def get_multi_roi(image, landmarks):
	'''obtain face, face mask, left and right eye ROI	'''

	# get face roi
	points = [36, 39, 42, 45, 48, 54]

	center_x = 0
	center_y = 0
	for i in points:
		center_x += landmarks[0, i]
		center_y += landmarks[1, i]
		#cv2.circle(image, (landmarks[0, i], landmarks[1, i]), radius=5, color=(0, 0, 255), thickness=-1)

	center_x /= 6
	center_y /= 6

	rect_w = int((center_x - landmarks[0, 0]) * 1.5)

	top_l_x = int(center_x - rect_w)
	top_l_y = int(center_y - rect_w)
	down_r_x = int(center_x + rect_w)
	down_r_y = int(center_y + rect_w)
	
	if (down_r_x > image.shape[1]):
		down_r_x = image.shape[1]
	if (down_r_y > image.shape[0]):
		down_r_y = image.shape[0]
	if (top_l_x < 0):
		top_l_x = 0
	if (top_l_y < 0):
		top_l_y = 0

	face = image[top_l_y: down_r_y, top_l_x: down_r_x]

	# get face mask
	face_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
	cv2.fillPoly(face_mask, np.array([[(top_l_x, top_l_y), (down_r_x, top_l_y), (down_r_x, down_r_y), (top_l_x, down_r_y)]], dtype=np.int32), color=1.0)

	# get eyes roi
	eye_width = 2 * max(abs(landmarks[0, points[0]] - landmarks[0, points[1]]), abs(landmarks[0, points[2]] - landmarks[0, points[3]]))
	eye_height = 1.0 * eye_width
	eye_middle_right = ((landmarks[0, points[0]] + landmarks[0, points[1]]) / 2, (landmarks[1, points[0]] + landmarks[1, points[1]]) / 2)
	eye_middle_left = ((landmarks[0, points[2]] + landmarks[0, points[3]]) / 2, (landmarks[1, points[2]] + landmarks[1, points[3]]) / 2)
	top_l_x = abs(int(eye_middle_left[0] - eye_width / 2.0))
	top_l_y = abs(int(eye_middle_left[1] - eye_height / 2.0))
	down_r_x = abs(int(eye_middle_left[0] + eye_width / 2.0))
	down_r_y = abs(int(eye_middle_left[1] + eye_height / 2.0))
	
	# segment eye image
	left_eye = image[top_l_y: down_r_y, top_l_x: down_r_x]

	top_l_x = abs(int(eye_middle_right[0] - eye_width / 2.0))
	top_l_y = abs(int(eye_middle_right[1] - eye_height / 2.0))
	down_r_x = abs(int(eye_middle_right[0] + eye_width / 2.0))
	down_r_y = abs(int(eye_middle_right[1] + eye_height / 2.0))
		
	right_eye = image[top_l_y : down_r_y, top_l_x : down_r_x]

	cv2.imshow('face_mask', face_mask)
	cv2.imshow('face', face)
	cv2.imshow('left_eye', left_eye)
	cv2.imshow('right_eye', right_eye)

	cv2.waitKey(1)

#	cv2.imwrite('temp/face_mask.jpg', face_mask)
#	cv2.imwrite('temp/face.jpg', face)
#	cv2.imwrite('temp/left_eye.jpg', left_eye)
#	cv2.imwrite('temp/right_eye.jpg', right_eye)

	return face, face_mask, left_eye, right_eye	
