# -*- coding: utf-8 -*-
'''
FileName: main.py
Author:   Wenyu
Date:     07/03/2019
Version:  v1.0 [07/03/2019][Wenyu] obtain face and landmarks with YOLO and SBR 
                                   by Tianlei
          v2.0 [07/04/2019][Wenyu] combine code and obtain gaze point by Shifan
          v2.1 [07/04/2019][Wenyu] format the code
          v2.2 [07/11/2019][Wenyu] formal the code on github
		  v2.3 [07/12/2019][Wenyu] add and test face detector
'''

#import yolo_video
#from util import get_multi_roi
#from gaze import test_small
#from gaze import test_mp2

from face_detect import detect_face
from landmark_detect import detect_landmarks

import cv2
import numpy as np

video_path = 'data/test_video.mp4'
image_path = 'data/test_image.jpg'


def main(input_type):
	'''
	main function
	'''
	if input_type == 'video':
		video_reader = cv2.VideoCapture(video_path)
	elif input_type == 'camera':
		video_reader = cv2.VideoCapture(0)
		video_reader.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		video_reader.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	else:
		frame = cv2.imread(image_path)

	#yolo = yolo_video.yolo_video()
	#track = test_mp2.test_small()
	face_detector = detect_face.FaceDetect()
	landmark_detector = detect_landmarks.LandmarkDetect()

	while True:
		# get image
		if input_type == 'video' or input_type == 'camera':
			ret, image = video_reader.read()
			if ret == False:
				break
		else:
			image = frame

		# get face box
		img, face = face_detector.detect_face(image)

		# get landmarks
		landmark_detector.detect_landmarks(img, face, image)
		
		#dic = yolo.detect(image)
		#if dic == None:
		#	continue

		#face_rect = dic['face']
		#landmarks = dic['landmarks']

		# segment roi
		#face, face_mask, left_eye, right_eye = get_multi_roi.get_multi_roi(image, landmarks)

		# predict gaze point
		#track.predict(face, face_mask, left_eye, right_eye)


if __name__ == '__main__':
	main('image')

