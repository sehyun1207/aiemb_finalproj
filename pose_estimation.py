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
"""Main script to run pose classification and pose estimation."""
import argparse
import logging
import sys
import time

import cv2
from ml import Classifier
from ml import Movenet
from ml import MoveNetMultiPose
from ml import Posenet
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import utils
import socket
import sys

def run(estimation_model: str, tracker_type: str, classification_model: str,
        label_file: str, camera_id: int, width: int, height: int, host:str, port:str) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    estimation_model: Name of the TFLite pose estimation model.
    tracker_type: Type of Tracker('keypoint' or 'bounding_box').
    classification_model: Name of the TFLite pose classification model.
      (Optional)
    label_file: Path to the label file for the pose classification model. Class
      names are listed one name per line, in the same order as in the
      classification model output. See an example in the yoga_labels.txt file.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
  """
  isconnect = False
  if host!="0" and port!="0":
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((host,int(port)))
    isconnect = True
    print("connect to server : %s %d",host,port)
  else:
    print("connect refused")

  # Initialize the pose estimator selected.
  if estimation_model in ['movenet_lightning', 'movenet_thunder', 'movenet_thunder_ptq_edgetpu']:
    pose_detector = Movenet(estimation_model)
  else:
    sys.exit('ERROR: Model is not supported.')

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  classification_results_to_show = 3
  fps_avg_frame_count = 10
  keypoint_detection_threshold_for_classifier = 0.1
  classifier = None

  # Initialize the classification model
  if classification_model:
    classifier = Classifier(classification_model,label_file)
    classification_results_to_show = min(classification_results_to_show,
                                         len(classifier.pose_class_names))

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    list_persons = [pose_detector.detect(image)]

    # Draw keypoints and edges on input image
    image = utils.visualize(image, list_persons)

    if classifier:
      # Check if all keypoints are detected before running the classifier.
      # If there's a keypoint below the threshold, show an error.
      person = list_persons[0]

      # Run pose classification
      prob_list = classifier.classify_pose(person)
      for i in range(1):
          class_name = prob_list[i].label
          probability = round(prob_list[i].score, 2)
          result_text = class_name + ' (' + str(probability) + ')'
          text_location = (left_margin, (i + 2) * row_size)
          cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                      font_size, text_color, font_thickness)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    if isconnect:
      msg_bytes = class_name.encode('utf-8')
      s.send(msg_bytes)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow(estimation_model, image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of estimation model.',
      required=False,
      default='movenet_lightning')
  parser.add_argument(
      '--tracker',
      help='Type of tracker to track poses across frames.',
      required=False,
      default='bounding_box')
  parser.add_argument(
      '--classifier', help='bool value', required=False, default='customed_model')
  parser.add_argument(
      '--label_file',
      help='Label file for classification.',
      required=False,
      default='labels.txt')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  parser.add_argument(
      '--host', help='server host', required=False, default='0')
  parser.add_argument(
      '--port', help='server port', required=False, default='0')

  args = parser.parse_args()

  run(args.model, args.tracker, args.classifier, args.label_file,
      int(args.cameraId), args.frameWidth, args.frameHeight, args.host, args.port)


if __name__ == '__main__':
  main()
