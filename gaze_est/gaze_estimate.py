"""Run inference a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

import argparse
import os
import sys

import tensorflow as tf

#import gaze_estimate_model
#from utils import preprocessing
#from utils import dataset_util
from gaze_est import gaze_estimate_model
from gaze_est.utils import preprocessing
from gaze_est.utils import dataset_util

import numpy as np
import timeit

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug

class GazeEstimate:

	def __init__(self):

		self.model_dir = 'E:/Collection/Study/EyeTracker/gaze_est/model'

		#self.parser = argparse.ArgumentParser()

		#self.parser.add_argument('--data_dir', type=str, default="/home/insfan/eye-tracker/MPIIFaceGaze_fem64/MPIIFaceGaze_fem64_p00.npz", help='The directory containing the image data.')

		#self.parser.add_argument('--output_dir', type=str, default='./dataset/inference_output',
		#		            help='Path to the directory to generate the inference results')

		#self.parser.add_argument('--infer_data_list', type=str, default='./dataset/sample_images_list.txt',
		#		            help='Path to the file listing the inferring images.')

		#self.parser.add_argument('--model_dir', type=str, default='E:/Collection/Study/EyeTracker/gaze_est/model',
		#		            help="Base directory for the model. "
		#		                 "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
		#		                 "with checkpoint name.")

		#self.parser.add_argument('--debug', action='store_true',
		#		            help='Whether to use debugger to track down bad values during training.')

		self._NUM_CLASSES = 21

		self._MEAN_LE = 0.47839513
		self._MEAN_RE = 0.2806009
		self._MEAN_F = 0.3976589

		self.pred = []


	# Import data
	def load_data(self, file):
		#npzfile = np.load(file)
		# face = np.array(f['faceData'])
		#self.train_eye_left = np.array(npzfile["train_eye_left"])
		#self.train_eye_right = np.array(npzfile["train_eye_right"])
		#self.train_face = np.array(npzfile["train_face"])
		#self.train_face_mask = np.array(npzfile["train_face_mask"])

		#print(train_eye_left.shape)

		self.test_eye_left = cv2.resize(self.test_eye_left, (64, 64))
		self.test_eye_right = cv2.resize(self.test_eye_right, (64, 64))
		self.test_face = cv2.resize(self.test_face, (64, 64))
		self.test_face_mask = cv2.resize(self.test_face_mask, (25, 25))

		#print(self.test_eye_left.shape)

		self.test_eye_left = np.reshape(self.test_eye_left, (1, ) + self.test_eye_left.shape)
		self.test_eye_right = np.reshape(self.test_eye_right, (1, ) + self.test_eye_right.shape)
		self.test_face = np.reshape(self.test_face, (1, ) + self.test_face.shape)
		self.test_face_mask = np.reshape(self.test_face_mask, (1, ) + self.test_face_mask.shape + (1, ))

		self.test_face_mask = self.test_face_mask.astype('float32') / 255.

		self.test_y = np.zeros((1, 1, 2, 1))

		#train_y = np.array(npzfile["train_y"])/10.0
		#train_y = np.squeeze(train_y)
		#return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y]
		return [self.test_eye_left, self.test_eye_right, self.test_face, self.test_face_mask, self.test_y]


	def normalize(self, data):
		shape = data.shape
		data = np.reshape(data, (shape[0], -1))
		data = data.astype('float32') / 255. # scaling
		data = data - np.mean(data) # normalizing
		return np.reshape(data, shape)


	def prepare_data_mp2(self, data):
		eye_left, eye_right, face, face_mask, y = data
		eye_left = eye_left.astype('float32') / 255. - self._MEAN_LE
		eye_right = eye_right.astype('float32') / 255. - self._MEAN_RE
		face = face.astype('float32') / 255. - self._MEAN_F
		face_mask = face_mask.astype('float32')
		# face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
		y = y.astype('float32')
		return [eye_left, eye_right, face, face_mask, y]


	def eval_input_fn(self, dataset, batch_size):
		"""Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

		Args:
			is_training: A boolean denoting whether the input is for training.
			data_dir: The directory containing the input data.
			batch_size: The number of samples per batch.
			num_epochs: The number of epochs to repeat the dataset.

		Returns:
			A tuple of images and labels.
		"""
		nums = dataset[0].shape[0]
		dataset0 = tf.convert_to_tensor(dataset[0])
		dataset1 = tf.convert_to_tensor(dataset[1])
		dataset2 = tf.convert_to_tensor(dataset[2])
		dataset3 = tf.convert_to_tensor(dataset[3])
		dataset4 = tf.convert_to_tensor(dataset[4])
		dataset = tf.data.Dataset.from_tensor_slices(((dataset0, dataset1, dataset2, dataset3), dataset4))

		# dataset = dataset.map(parse_record)
		# dataset = dataset.map(
		#     lambda image, label: preprocess_image(image, label, is_training))
		dataset = dataset.prefetch(batch_size)

		# We call repeat after shuffling, rather than before, to prevent separate
		# epochs from blending together.
		dataset = dataset.batch(batch_size)

		iterator = dataset.make_one_shot_iterator()
		images, labels = iterator.get_next()

		return images, labels


	def mains(self, unused_argv):
		# Using the Winograd non-fused algorithms provides a small performance boost.
		os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
		#pred_hooks = None
		#if self.FLAGS.debug:
		#	debug_hook = tf_debug.LocalCLIDebugHook()
		#	pred_hooks = [debug_hook]

		#dataset = self.load_data(self.FLAGS.data_dir)
		dataset = self.load_data('')
		dataset = self.prepare_data_mp2(dataset)
		# dataset = shuffle_data(dataset)
		nums = dataset[-1].shape[0]
		val_data = [dataset[0][int(0.8*nums):,:,:,:], dataset[1][int(0.8*nums):,:,:,:], dataset[2][int(0.8*nums):,:,:,:], dataset[3][int(0.8*nums):,:,:,:], dataset[4][int(0.8*nums):,:,:,:]]

		# model = tf.estimator.Estimator(
		#     model_fn=gaze_estimate_model.gaze_estimate_model_fn,
		#     model_dir=FLAGS.model_dir,
		#     params={
		#         'batch_size': 1,  # Batch size must be 1 because the images' size may differ
		#         'batch_norm_decay': None,
		#     })
		# predictions = model.predict(
		#       input_fn=lambda: eval_input_fn(val_data, 1),
		#       hooks=pred_hooks)
		#
		# # # Manually load the latest checkpoint
		# # saver = tf.train.Saver()
		# # with tf.Session() as sess:
		# #     ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
		# #     saver.restore(sess, ckpt.model_checkpoint_path)
		# #     preds = sess.run(predictions)
		# output_dir = FLAGS.output_dir
		# if not os.path.exists(output_dir):
		#   os.makedirs(output_dir)
		# zip(predictions)
		# print(predictions['coordinate'])
		# np.savetxt(output_dir + '/gt.txt', (np.squeeze(val_data[-1])))

		# np.save(output_dir+'/predict.txt', (predictions))

		features, labels = self.eval_input_fn(val_data, 1)

		predictions = gaze_estimate_model.gaze_estimate_model_fn(
			features,
			labels,
			tf.estimator.ModeKeys.EVAL,
			params={
				'batch_size': 1,  # Batch size must be 1 because the images' size may differ
				'batch_norm_decay': None,
				'freeze_batch_norm': True
			}).predictions

		# Manually load the latest checkpoint
		print('### D1')
		saver = tf.train.Saver()
		print('### D2')
		pred  = []
		with tf.Session() as sess:
			print('### D3')
			#ckpt = tf.train.get_checkpoint_state(self.FLAGS.model_dir)
			ckpt = tf.train.get_checkpoint_state(self.model_dir)
			print('### D4')

			print('### D4.1 ', ckpt.model_checkpoint_path)

			saver.restore(sess, ckpt.model_checkpoint_path)
			print('### D5')
			# Loop through the batches and store predictions and labels
			step = 1
			start = timeit.default_timer()
			while True:
				try:
					preds = sess.run(predictions)
					pred.append(preds['coordinate'].tolist())
					if not step % 100:
						stop = timeit.default_timer()
						tf.logging.info("current step = {} ({:.3f} sec)".format(step, stop - start))
						start = timeit.default_timer()
					step += 1
					#self.pred = pred
					#print(self.pred)

				except tf.errors.OutOfRangeError:
					break

		#output_dir = self.FLAGS.output_dir
		#if not os.path.exists(output_dir):
		#	os.makedirs(output_dir)

		#print(pred)
		return pred[0]

		#  np.savetxt(output_dir + '/gt.txt', (np.squeeze(val_data[-1])*10))
		#  np.savetxt(output_dir+'/predict.txt', np.squeeze(pred))

		#  np.savetxt(output_dir+'/gt_predict.txt', np.concatenate((np.squeeze(val_data[-1])*10, np.squeeze(pred) *10), axis=1))


	def predict(self, face, face_mask, left_eye, right_eye):
		self.test_face = face
		self.test_face_mask = face_mask
		self.test_eye_left = left_eye
		self.test_eye_right = right_eye

		tf.logging.set_verbosity(tf.logging.INFO)
		#self.FLAGS, unparsed = self.parser.parse_known_args()
		#print('argv = ', [sys.argv[0]] + unparsed)
		return self.mains([sys.argv[0]])
		#tf.app.run(main=self.mains, argv=[sys.argv[0]] + unparsed)


#if __name__ == '__main__':
#	test_eye_left = cv2.imread('data/test_left_eye.jpg')
#	test_eye_right = cv2.imread('data/test_right_eye.jpg')
#	test_face = cv2.imread('data/test_face.jpg')
#	test_face_mask = cv2.imread('data/test_face_mask.jpg', 0)

#	gE = GazeEstimate()
#	pred = gE.predict(test_face, test_face_mask, test_eye_left, test_eye_right)
#	print(pred)

