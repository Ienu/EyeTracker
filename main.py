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
		  v2.3 [07/12/2019][Wenyu] add and test face detector and landmark detector
		  v2.4 [07/15/2019][Wenyu] add gaze estimator and try except
'''

from face_detect import detect_face
from landmark_detect import detect_landmarks
from gaze_estimate import gaze_estimation

from utils.get_multi_roi import get_multi_roi

import cv2
import numpy as np

from PIL import Image

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

	landmark_detector = detect_landmarks.LandmarkDetect()
	face_detector = detect_face.FaceDetect()
	gaze_estimator = gaze_estimation.GazeEstimate()

	while True:
		# get image
		if input_type == 'video' or input_type == 'camera':
			ret, image = video_reader.read()
			if ret == False:
				break
		else:
			image = cv2.imread(image_path)

		try:
			# get face box
			face_box = face_detector.detect_face(image)

			if face_box == None:
				continue

			# get landmarks
			img = Image.fromarray(image)
			landmarks = landmark_detector.detect_landmarks(img, face_box)

			# show face rect and landmarks
			image_show = image.copy()
			cv2.rectangle(image_show, (face_box[1], face_box[2]), (face_box[3], face_box[4]), color=(0, 255, 0))
			for j in range(landmarks.shape[1]):
				point = landmarks[:, j]
				cv2.circle(image_show, (int(point[0]), int(point[1])), radius=5,color=(0, 0, 255), thickness=-1)
			
			# segment roi
			face, face_mask, left_eye, right_eye = get_multi_roi(image, landmarks)

			# predict gaze point
			pred = gaze_estimator.predict(face, face_mask, left_eye, right_eye)
			print(pred)

			image_show = cv2.flip(image_show, 1)
			cv2.circle(image_show, (int(pred[0]), int(pred[1])), radius=10, color=(255, 255, 255), thickness=5)
			cv2.imshow('dst', image_show)
			cv2.waitKey(1)

		except:
			if input_type == 'camera':
				video_reader.release()
			break

if __name__ == '__main__':
	main('camera')

