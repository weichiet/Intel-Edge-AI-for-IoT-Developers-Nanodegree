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
import math
sys.path.insert(0, str(Path().resolve().parent.parent))

'''
Gaze Estimation Model
https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
'''

class GazeEstimation:
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

        self.input_head_pose_angles = None
        self.input_left_eye = None
        self.input_right_eye = None

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
        #Load the model as IR
        self.network = IENetwork(model=model_structure, weights=model_weight)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load(self.network)
        self.loading_time = time.time() - start_model_load_time

        # Get the input layer
        self.input_head_pose_angles = self.network.inputs['head_pose_angles']

        self.output_blob = next(iter(self.network.outputs))


    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        This method is meant for running predictions on the input image.
        '''

        self.initial_width = left_eye_image.shape[1]
        self.initial_height = left_eye_image.shape[0]
        left_eye_image,right_eye_image = self.preprocess_input(left_eye_image,right_eye_image)

        start_inference_time = time.time()
        # self.exec_network.requests[0].async_infer(inputs={self.input_blob: {"left_eye_image":left_eye_image,"right_eye_image":right_eye_image}})
        self.exec_network.requests[0].async_infer(inputs={"head_pose_angles": head_pose_angles,
                "left_eye_image":left_eye_image,
                "right_eye_image":right_eye_image})

        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_blob]
            out = self.preprocess_output( outputs)
            self.inference_time.append(time.time() - start_inference_time)

            return out

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, left_eye_image, right_eye_image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''

        left_eye_image = cv2.resize(left_eye_image, (60, 60))
        left_eye_image = left_eye_image.transpose((2,0,1))
        left_eye_image = left_eye_image.reshape((1, 3, 60, 60))

        right_eye_image = cv2.resize(right_eye_image, (60, 60))
        right_eye_image = right_eye_image.transpose((2,0,1))
        right_eye_image = right_eye_image.reshape((1, 3, 60, 60))

        return left_eye_image,right_eye_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        #The net outputs a blob with the shape: [1, 3], containing Cartesian
        #   coordinates of gaze direction vector. Please note that the output
        #   vector is not normalizes and has non-unit length.
        #   Output layer name in Inference Engine format: 'gaze_vector'

        gaze_vector = outputs[0]
        roll = gaze_vector[2]#pose_angles[0][2][0]
        cs = math.cos(roll * math.pi / 180.0)
        sn = math.sin(roll * math.pi / 180.0)

        tmpX = gaze_vector[0] * cs + gaze_vector[1] * sn
        tmpY = -gaze_vector[0] * sn + gaze_vector[1] * cs

        return (tmpX,tmpY), (gaze_vector)

    def get_total_inference_time(self):
        return sum(np.array(self.inference_time, dtype=np.float))
