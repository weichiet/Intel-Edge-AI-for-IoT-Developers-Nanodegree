import os
import sys
import json
import time
import cv2

from threading import Thread
from collections import namedtuple
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
import logging as log
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FaceLandmarksDetection
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController


model_dir = '../models/intel/'
model_dir_face = {
    'FP32': model_dir + 'face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml',
    'FP16': model_dir + 'face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml',
    'INT8': model_dir + 'face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml'
}
model_dir_landmarks = {
    'FP32': model_dir + 'landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml',
    'FP16': model_dir + 'landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml',
    'INT8': model_dir + 'landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml'
}
model_dir_pose = {
    'FP32': model_dir + 'head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml',
    'FP16': model_dir + 'head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml',
    'INT8': model_dir + 'head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml'
}
model_dir_gaze = {
    'FP32': model_dir + 'gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml',
    'FP16': model_dir + 'gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml',
    'INT8': model_dir + 'gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml'
}

def args_parser():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-p","--precision", default='FP16',
                        help='Precision of models(FP32, FP16 or INT8).'
                             'Each model in the pipeline will be used with this precision if available.\n'
                             'Default: FP16')
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image or 'cam' for capturing video stream from camera")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.")
    parser.add_argument("-t", "--threshold", default=0.5, type=float,
                        help="Probability threshold for detections filtering")
    parser.add_argument("-o", "--output_dir", help = "Path to output directory", type=str, default=None)
    parser.add_argument("-di", "--display_intermediate", default=None, type=str,
                        help="Select between yes | no ")
    return parser

def main(args):

    # Initialize mouse controller
    controller = MouseController("medium", "fast")

    # Initialize logging function
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)

    # Check input type
    #args = args_parser().parse_args()
    if args.input == 'cam':
       input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    # Initialize input feeder
    cap = cv2.VideoCapture(input_stream)
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = os.path.join(args.output_dir, args.device, args.precision)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Initialize final output video writer
    out_final = cv2.VideoWriter(os.path.join(output_path, "output.mp4"),
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)

    # Initialize intermediate output video writer
    if args.display_intermediate == 'yes':
        out_face = cv2.VideoWriter(os.path.join(output_path, "output_face.mp4"),
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
        out_landmarks = cv2.VideoWriter(os.path.join(output_path, "output_landmarks.mp4"),
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
        out_pose = cv2.VideoWriter(os.path.join(output_path, "output_pose.mp4"),
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
        out_gaze = cv2.VideoWriter(os.path.join(output_path, "output_gaze.mp4"),
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)

    # Open image or video source
    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video/image source")
        return

    # Initialise the model classes
    if args.cpu_extension:
        model_face = FaceDetection(model_dir_face[args.precision], args.device, args.threshold, args.cpu_extension)
        model_pose = HeadPoseEstimation(mmodel_dir_pose[args.precision], args.device, args.threshold, args.cpu_extension)
        model_landmarks = FaceLandmarksDetection(model_dir_landmarks[args.precision], args.device, args.threshold, args.cpu_extension)
        model_gaze = GazeEstimation(model_dir_gaze[argsprecision], args.device, args.threshold, args.cpu_extension)
    else:
        model_face = FaceDetection(model_dir_face[args.precision], args.device, args.threshold)
        model_pose = HeadPoseEstimation(model_dir_pose[args.precision], args.device, args.threshold)
        model_landmarks = FaceLandmarksDetection(model_dir_landmarks[args.precision], args.device, args.threshold)
        model_gaze = GazeEstimation(model_dir_gaze[args.precision], args.device, args.threshold)

    # Load the network to IE plugin
    model_face.load_model()
    model_pose.load_model()
    model_landmarks.load_model()
    model_gaze.load_model()

    print("All models are loaded successfully")

    frame_count = 0
    print("Start inference....")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print ("End of Frame: BREAKING")
            break

        frame_count += 1

        if frame is None:
            log.error("Blank frame grabbed")
            break

        # Detect face in frame
        coords, frame = model_face.predict(frame)

        # Dislay face detection intermediate result
        if args.display_intermediate == 'yes':
            out_face.write(frame)

        # If a face is detected in frame
        if len(coords) > 0:
            # Get coordinates of first detected face
            [xmin,ymin,xmax,ymax] = coords[0]
            # Crop the image of the head
            head_pose = frame[ymin:ymax, xmin:xmax]

            # Estimate head pose angles
            pose_angles = model_pose.predict(head_pose)

            #  Dislay head pose estimation intermediate result
            if args.display_intermediate == 'yes':
                p = "Pose Angles {}".format(pose_angles)
                cv2.putText(frame, p, (50, 15), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (255,0, 0), 3)
                out_pose.write(frame)

            # Detect facial landmarks
            coords, f = model_landmarks.predict(head_pose)
            frame[ymin:ymax, xmin:xmax] = f

            # Dislay facial landmarks detection intermediate result
            if args.display_intermediate == "yes":
                out_landmarks.write(frame)

            # Crop left eye and right eye
            [[xlmin,ylmin,xlmax,ylmax],[xrmin,yrmin,xrmax,yrmax]] = coords
            left_eye_image = f[ylmin:ylmax, xlmin:xlmax]
            right_eye_image = f[yrmin:yrmax, xrmin:xrmax]

            # Estimate gaze direction
            output, gaze_vector = model_gaze.predict(left_eye_image, right_eye_image, pose_angles)

            # Dislay gaze estimation intermediate result
            if args.display_intermediate == 'yes':
                p = "Gaze Vector {}".format(gaze_vector)
                cv2.putText(frame, p, (50, 15), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (255, 0, 0), 1)
                fl = draw_gaze(left_eye_image, gaze_vector)
                fr = draw_gaze(right_eye_image, gaze_vector)
                f[ylmin:ylmax, xlmin:xlmax] = fl
                f[yrmin:yrmax, xrmin:xrmax] = fr
                out_gaze.write(frame)

            # Move the mouse pointer every 10 frame
            if frame_count%10 == 0:
                controller.move(output[0], output[1])

        # Calculate the performance
        total_time = model_face.get_total_inference_time() + \
                        model_pose.get_total_inference_time() + \
                        model_landmarks.get_total_inference_time() + \
                        model_gaze.get_total_inference_time()
        total_inference_time=round(total_time, 1)
        fps=frame_count/total_inference_time

        print(f"Total_inference_time:{total_inference_time}, FPS:{fps:.3}")

        out_final.write(frame)

    print("Finish inference....")
    total_model_load_time = model_face.loading_time + \
                                model_pose.loading_time + \
                                model_landmarks.loading_time + \
                                model_gaze.loading_time

    # Write the stats files
    with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
        f.write(str(total_inference_time)+'\n')
        f.write(str(fps)+'\n')
        f.write(str(total_model_load_time)+'\n')

    # Release cv2 cap
    cap.release()
    cv2.destroyAllWindows()

    # Release all video writers
    out_final.release()
    if args.display_intermediate == 'yes':
        out_face.release()
        out_pose.release()
        out_landmarks.release()
        out_gaze.release()


def draw_gaze(screen_img, gaze_pts, gaze_colors=None, scale=4, return_img=False, cross_size=16, thickness=10):

    """ Draws an "x"-shaped cross on a screen for given gaze points, ignoring missing ones
    """
    width = int(cross_size * scale)

    draw_cross(screen_img, gaze_pts[0] * scale, gaze_pts[1] * scale,
        (0, 0, 255), width, thickness)
    return  screen_img

def draw_cross(bgr_img,x, y,color=(255, 255, 255), width=2, thickness=0.5):

    """ Draws an "x"-shaped cross at (x,y)
    """
    x, y, w = int(x), int(y), int(width / 2)  # ensure points are ints for cv2 methods

    cv2.line(bgr_img, (x - w , y - w), (x + w , y + w), color, thickness)
    cv2.line(bgr_img, (x - w , y + w), (x + w, y - w), color, thickness)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-p","--precision", default='FP16',
                        help='Precision of models(FP32, FP16 or INT8).'
                             'Each model in the pipeline will be used with this precision if available.\n'
                             'Default: FP16')
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image or 'cam' for capturing video stream from camera")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.")
    parser.add_argument("-t", "--threshold", default=0.5, type=float,
                        help="Probability threshold for detections filtering")
    parser.add_argument("-o", "--output_dir", default='../bin/', type=str,
                        help = "Path to output directory")
    parser.add_argument("-di", "--display_intermediate", default='no', type=str,
                        help="Select between 'yes' | 'no' to display intermediate output")

    parsed_args = parser.parse_args()

    main(parsed_args)
