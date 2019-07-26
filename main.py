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
		  v2.5 [07/18/2019][Wenyu] add time measurement
          v2.6 [07/26/2019][Wenyu] add multi modes and deal with dataset
'''

from face_detect import detect_face
from landmark_detect import detect_landmarks
from gaze_estimate import gaze_estimation

from utils.get_multi_roi import get_multi_roi
from utils.dataset_preprocess import dataset_preprocess

import cv2
import numpy as np
import time

from PIL import Image

video_path = 'data/test_video.mp4'
image_path = 'data/test_image.jpg'
dataset_path = 'data/MPIIFaceGaze'
#dataset_path = 'data/test'

save_name = 'mpii_p00.npz'


def main(input_type, gaze_mode=0, landmark_mode=0, face_mode=0, tim=False, save_before_gaze=True):
	'''
	main function
	input_type: vide, camera, image, dataset
	mode: 0->predict, -1->disabled, 1->train
	'''
	if input_type == 'video':
		video_reader = cv2.VideoCapture(video_path)
	elif input_type == 'camera':
		video_reader = cv2.VideoCapture(0)
		video_reader.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		video_reader.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	elif input_type == 'dataset':
		data_generator = dataset_preprocess(dataset_path)

	if landmark_mode == 0:
		landmark_detector = detect_landmarks.LandmarkDetect()
	
	if face_mode == 0:	
		face_detector = detect_face.FaceDetect()
	
	if gaze_mode == 0:
		gaze_estimator = gaze_estimation.GazeEstimate()

	if save_before_gaze == True:
		save_data = {}
		save_data['face'] = np.zeros((0, 64, 64, 3))
		save_data['face_mask'] = np.zeros((0, 64, 64))
		save_data['left_eye'] = np.zeros((0, 64, 64, 3))
		save_data['right_eye'] = np.zeros((0, 64, 64, 3))
		save_data['gaze_point'] = np.zeros((0, 2))

	print('init finish...')

	while True:
		# get image
		if input_type == 'video' or input_type == 'camera':
			ret, image = video_reader.read()
			if ret == False:
				break
		elif input_type == 'image':
			image = cv2.imread(image_path)
		elif input_type == 'dataset':
			try:
				image, gaze_x, gaze_y = next(data_generator)
			except StopIteration:
				print('dataset iteration finish...')
				np.savez_compressed(save_name, 
					face=save_data['face'], 
					face_mask=save_data['face_mask'],
					left_eye=save_data['left_eye'],
					right_eye=save_data['right_eye'],
					gaze_point=save_data['gaze_point'])
				break
		else:
			print('input_type error')
			break

		try:
			# get face box
			if tim == True: # consider to use decorator
				face_ts = time.time()

			if face_mode == 0:
				face_box = face_detector.detect_face(image)

			if tim == True:
				face_te = time.time()

			if face_box == None:
				continue

			#print('get face box...')

			# get landmarks
			#if image == None:
			#	print('no image...')
			#	break
			
			img = Image.fromarray(image)

			if tim == True:
				landmark_ts = time.time()

			if landmark_mode == 0:
				landmarks = landmark_detector.detect_landmarks(img, face_box)

			if tim == True:
				landmark_te = time.time()

			# show face rect and landmarks
			image_show = image.copy()
			cv2.rectangle(image_show, (face_box[1], face_box[2]), (face_box[3], face_box[4]), color=(0, 255, 0))
			for j in range(landmarks.shape[1]):
				point = landmarks[:, j]
				cv2.circle(image_show, (int(point[0]), int(point[1])), radius=5,color=(0, 0, 255), thickness=-1)
			
			#print('get landmarks done...')

			# segment roi
			face, face_mask, left_eye, right_eye = get_multi_roi(image, landmarks)

			if save_before_gaze == True:
				_face = cv2.resize(face, (64, 64))
				_face = np.expand_dims(_face, axis=0)

				_face_mask = cv2.resize(face_mask, (64, 64))
				_face_mask = np.expand_dims(_face_mask, axis=0)
				#print(_face_mask.shape)

				_left_eye = cv2.resize(left_eye, (64, 64))
				_left_eye = np.expand_dims(_left_eye, axis=0)

				_right_eye = cv2.resize(right_eye, (64, 64))
				_right_eye = np.expand_dims(_right_eye, axis=0)

				_gaze = np.expand_dims(np.array([gaze_x, gaze_y]), axis=0)
				#_gaze = np.array([gaze_x, gaze_y])
				#print(_gaze.shape)

				save_data['face'] = np.append(save_data['face'], _face, axis=0)
				save_data['face_mask'] = np.append(save_data['face_mask'], _face_mask, axis=0)
				save_data['left_eye'] = np.append(save_data['left_eye'], _left_eye, axis=0)
				save_data['right_eye'] = np.append(save_data['right_eye'], _right_eye, axis=0)
				save_data['gaze_point'] = np.append(save_data['gaze_point'], _gaze, axis=0) 

			# predict gaze point
			if tim == True:
				gaze_ts = time.time()

			if gaze_mode == 0:
				pred = gaze_estimator.predict(face, face_mask, left_eye, right_eye)
					
			if tim == True:
				gaze_te = time.time()

			#print(pred)
			if tim == True:
				print('face detect time: ', face_te - face_ts)
				print('landmark detect time: ', landmark_te - landmark_ts)
				print('gaze estimation time: ', gaze_te - gaze_ts)

			if False:#pred != None:
				image_show = cv2.flip(image_show, 1)
				cv2.circle(image_show, (int(pred[0]), int(pred[1])), radius=10, color=(255, 255, 255), thickness=5)
			cv2.imshow('dst', image_show)
			cv2.waitKey(1)

		except:
			if input_type == 'camera':
				video_reader.release()
			if input_type == 'dataset':
				np.savez_compressed(save_name, 
					face=save_data['face'], 
					face_mask=save_data['face_mask'],
					left_eye=save_data['left_eye'],
					right_eye=save_data['right_eye'],
					gaze_point=save_data['gaze_point'])
			break

	if input_type == 'camera':
		video_reader.release()

if __name__ == '__main__':
	main('dataset', -1, 0, 0)

