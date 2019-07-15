"""Run inference a DeepLab v3 model using tf.estimator API."""
import os
import sys

import tensorflow as tf
import cv2
import numpy as np
import timeit

from PIL import Image

class GazeEstimate:

	def __init__(self):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    self.sess = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(self.sess, ["serve"], 'E:/Collection/Study/EyeTracker/gaze_est2/model/1563091393')
    graph = tf.get_default_graph()

  def predict(self, test_face, test_face_mask, test_left_eye, test_right_eye):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    eye_left = self.sess.graph.get_tensor_by_name('eye_left:0')
    eye_right = self.sess.graph.get_tensor_by_name('eye_right:0')
    face = self.sess.graph.get_tensor_by_name('face:0')
    face_mask = self.sess.graph.get_tensor_by_name('face_mask:0')
    pred = self.sess.graph.get_tensor_by_name('gaze_estimate/conv3/BiasAdd:0')

    test_left_eye = cv2.resize(test_left_eye, (64, 64))
    test_right_eye = cv2.resize(test_right_eye, (64, 64))
    test_face = cv2.resize(test_face, (64, 64))
    test_face_mask = cv2.resize(test_face_mask, (64, 64))

    test_left_eye = np.expand_dims(test_left_eye, axis=0)
    test_right_eye = np.expand_dims(test_right_eye, axis=0)
    test_face = np.expand_dims(test_face, axis=0)
    test_face_mask = np.expand_dims(test_face_mask, axis=0)
    test_face_mask = np.expand_dims(test_face_mask, axis=-1)

    preds = self.sess.run(pred, feed_dict={eye_left: test_left_eye, eye_right: test_right_eye, face: test_face, face_mask: test_face_mask})
    result = np.squeeze(preds)*10
    print("predict: ", result)
    return result
