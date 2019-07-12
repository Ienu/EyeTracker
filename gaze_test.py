from gaze_est import gaze_estimate

import cv2
import numpy as np

gaze_estimator = gaze_estimate.GazeEstimate()

test_eye_left = cv2.imread('data/test_left_eye.jpg')
test_eye_right = cv2.imread('data/test_right_eye.jpg')
test_face = cv2.imread('data/test_face.jpg')
test_face_mask = cv2.imread('data/test_face_mask.jpg', 0)

pred = gaze_estimator.predict(test_face, test_face_mask, test_eye_left, test_eye_right)
print(pred)
