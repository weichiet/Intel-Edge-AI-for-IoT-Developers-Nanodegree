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
Face Detection Model
https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
'''

class FaceDetection:
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
        #Load the model as IR
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
        #The net outputs blob with shape: [1, 1, N, 7], where N is the number
        #   of detected bounding boxes. For each detection, the description has
        #   the format: [image_id, label, conf, x_min, y_min, x_max, y_max]

        coords = []

        for obj in outputs[0][0]:
            # Draw bounding box for object when it's probability is more than
            #  the specified threshol
            if obj[2] > float(self.threshold):
                xmin = int(obj[3] * self.initial_width)
                ymin = int(obj[4] * self.initial_height)
                xmax = int(obj[5] * self.initial_width)
                ymax = int(obj[6] * self.initial_height)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                coords.append([xmin,ymin,xmax,ymax])
                break # Assuming one person looking at the camera

        return frame, coords

    def get_total_inference_time(self):
        return sum(np.array(self.inference_time, dtype=np.float))

    def get_total_load_time(self):
        return sum(np.array(self.loading_time, dtype=np.float))
