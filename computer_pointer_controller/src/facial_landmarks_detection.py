import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin
import os
import cv2
import argparse
import time
import sys
from argparse import ArgumentParser
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))

'''
Face Landmarks Detection Model
https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
'''

class FaceLandmarksDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.5,  extensions=None):
        '''
        Initialise instance variables.
        '''
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.output_blob = None
        self.threshold = threshold
        self.device = device
        self.model_name = model_name
        self.extensions = extensions
        self.initial_width = None
        self.initial_height = None
        self.loading_time = -1.0
        self.inference_time = []

    def load_model(self):
        '''
        Load the model to the device specified by the user.
        '''
        # Get the model structure and weights
        model_structure = self.model_name
        model_weight = os.path.splitext(model_structure)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IEPlugin(device=self.device)

        # Add a CPU extension, if applicable
        if self.extensions and "CPU" in self.device:
            self.plugin.add_cpu_extension(self.extensions)

        start_model_load_time = time.time()
        # Load the model as IR
        self.network = IENetwork(model=model_structure, weights=model_weight)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load(self.network)
        self.loading_time = time.time() - start_model_load_time

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))


    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.initial_width = image.shape[1]
        self.initial_height = image.shape[0]

        coords = None
        frame = self.preprocess_input(image)

        start_inference_time = time.time()
        # Async mode
        self.exec_network.requests[0].async_infer(inputs={self.input_blob: frame})

        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_blob]
            frame, coords = self.preprocess_output(image, outputs)
            self.inference_time.append(time.time() - start_inference_time)

            return coords, frame

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''

        (n, c, h, w) = self.network.inputs[self.input_blob].shape
        frame = cv2.resize(image, (w, h))
        frame = frame.transpose((2,0,1))
        frame = frame.reshape((n, c, h, w))

        return frame

    def preprocess_output(self, frame, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        #The net outputs a blob with the shape: [1, 10],
        #   containing a row-vector of 10 floating point values for
        #   five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
        #   All the coordinates are normalized to be in range [0,1].

        box_width = 20
        box_height = 20

        coords = []
        outputs= outputs[0]

        # Get the coordinates of left eye and right eye
        x_left, y_left = outputs[0][0]*self.initial_width, outputs[1][0]*self.initial_height
        x_right, y_right = outputs[2][0]*self.initial_width, outputs[3][0]*self.initial_height

        # Coordinates of left eye bounding box
        x_left_min = x_left - box_width
        y_left_min = y_left - box_height
        x_left_max = x_left + box_width
        y_left_max = y_left + box_height

        # Coordinates of right eye bounding box
        x_right_min = x_right - box_width
        y_right_min = y_right - box_height
        x_right_max = x_right + box_width
        y_right_max = y_right + box_height

        cv2.rectangle(frame, (x_left_min, y_left_min), (x_left_max, y_left_max), (0, 55, 255), 1)
        cv2.rectangle(frame, (x_right_min, y_right_min), (x_right_max, y_right_max), (0, 55, 255), 1)
        coords = [[int(x_left_min), int(y_left_min), int(x_left_max), int(y_left_max)],
                    [int(x_right_min), int(y_right_min), int(x_right_max), int(y_right_max)]]

        return frame, coords

    def get_total_inference_time(self):
        return sum(np.array(self.inference_time, dtype=np.float))
