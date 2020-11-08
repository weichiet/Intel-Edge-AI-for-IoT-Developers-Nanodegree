"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Handle image, video or webcam
    # Create a flag for single images
    image_flag = False
    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input= 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True

    # Get and open video capture
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    #in_shape = net_input_shape['image_tensor']

    #iniatilize variables
    current_count = 0
    counter = 0
    counter_prev = 0
    total_count = 0
    duration_prev = 0
    duration = 0

    NUM_FRAME = 5

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape['image_tensor'][3], net_input_shape['image_tensor'][2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        avg_duration = None
        start_time = time.time()
        infer_network.exec_net(net_input=p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            detection_time = time.time() - start_time
            result = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            num_detected = 0
            probs = result[0, 0, :, 2]
            for i, p in enumerate(probs):
                if p > prob_threshold:
                    num_detected += 1
                    box = result[0, 0, i, 3:]
                    xmin = int(box[0] * width)
                    ymin = int(box[1] * height)
                    xmax = int(box[2] * width)
                    ymax = int(box[3] * height)
                    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

            # Printing Inference Time
            inference_time = "Inference time: {:.3f}ms"\
                               .format(detection_time * 1000)
            cv2.putText(frame, inference_time, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

            # Count as valid detection if the person appears in more than `NUM_FRAME` continuous frames
            if num_detected != counter:
                counter_prev = counter
                counter = num_detected
                if duration >= NUM_FRAME:
                    duration_prev = duration
                    duration = 0
                else:
                    duration = duration_prev + duration
                    duration_prev = 0
            else:
                duration += 1
                if duration >= NUM_FRAME:
                    current_count = counter
                    if duration == NUM_FRAME and counter > counter_prev:
                        total_count += counter - counter_prev
                    elif duration == NUM_FRAME and counter < counter_prev:
                        # The person exit the scene, report the duration in ms (note: Input video with FPS=10)
                        avg_duration = int((duration_prev / 10.0) * 1000)

            client.publish('person',
                           payload=json.dumps({'count': current_count, 'total': total_count}),
                           qos=0, retain=False)

            if avg_duration is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': avg_duration}),
                               qos=0, retain=False)

            # Break if escape key pressed
            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        #frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
