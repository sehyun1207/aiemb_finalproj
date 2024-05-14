# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code to run a TFLite pose classification model."""
import os
from typing import List

from data import Category
from data import Person
import numpy as np

import math

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


STOP = 100
GO_AHEAD = 101
RIGHT = 102
LEFT = 103
GO_BACk = 104


bias = 20
# pylint: disable=g-import-not-at-top
try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  # from tflite_runtime.interpreter import Interpreter
  from pycoral.adapters import common
  from pycoral.utils.edgetpu import make_interpreter
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter
# pylint: enable=g-import-not-at-top


class Classifier(object):
    def __init__(self) -> None:
        self.name = 0
    def classify_pose(self, person: Person) -> List[Category]:
        # Initialize the label of the pose. It is not known at this stage.
        label = "STOP"

        input_tensor = [[keypoint.coordinate.y, keypoint.coordinate.x, keypoint.score] for keypoint in person.keypoints]


        # Calculate the required angles.
        #----------------------------------------------------------------------------------------------------------------

        # Get the angle between the left shoulder, elbow and wrist points.
        # 9번, 7번, 5번 landmark
        # 왼쪽 어깨, 왼쪽 팔꿈치, 왼쪽 손목 landmark angle 값 계산
        left_elbow_angle = calculateAngle(input_tensor[LEFT_SHOULDER],
                                          input_tensor[LEFT_ELBOW],
                                          input_tensor[LEFT_WRIST])

        # 6번, 8번, 10번 landmark
        # 오른쪽 어깨, 오른쪽 팔꿈치, 오른쪽 손목 landmark angle 값 계산
        right_elbow_angle = calculateAngle(input_tensor[RIGHT_SHOULDER],
                                          input_tensor[RIGHT_ELBOW],
                                          input_tensor[RIGHT_WRIST])

        # 7번, 5번, 6번 landmark
        # 왼쪽 팔꿈치, 왼쪽 어깨, 오른쪽 어깨 landmark angle 값 계산
        left_shoulder_angle = calculateAngle(input_tensor[LEFT_ELBOW],
                                          input_tensor[LEFT_SHOULDER],
                                          input_tensor[RIGHT_SHOULDER])

        # 5번, 6번, 4번 landmark
        # 왼쪽 팔꿈치, 왼쪽 어깨, 오른쪽 어깨 landmark angle 값 계산
        right_shoulder_angle = calculateAngle(input_tensor[LEFT_SHOULDER],
                                          input_tensor[RIGHT_SHOULDER],
                                          input_tensor[RIGHT_ELBOW])


        #----------------------------------------------------------------------------------------------------------------

        if left_elbow_angle>180-bias and left_elbow_angle<180+bias and right_elbow_angle>180-bias and right_elbow_angle<180+bias:
          if left_shoulder_angle>180-bias and left_shoulder_angle<180+bias and right_shoulder_angle>180-bias and right_shoulder_angle<180+bias:
              label="GO_AHEAD"
        elif left_elbow_angle>180-bias and left_elbow_angle<180+bias and right_shoulder_angle<270+bias:
              label="LEFT"
        elif right_elbow_angle>180-bias and right_elbow_angle<180+bias and left_shoulder_angle<270+bias :
              label="RIGHT"
        if left_shoulder_angle>90-bias and left_shoulder_angle<90+bias and right_shoulder_angle>90-bias and right_shoulder_angle<90+bias:
              label="GO_BACK"

        #----------------------------------------------------------------------------------------------------------------

        return label


def calculateAngle(landmark1, landmark2, landmark3):

    # Get the required landmarks coordinates.
    x1, y1,_ = landmark1
    x2, y2,_ = landmark2
    x3, y3,_ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle