"""
yolo_detection.py
Summary: Main script for UAV MobFinder. Handles video capture, object detection, and telemetry data recording.
TODO:
- Add support for multiple cameras
- Add support for multiple object detection models
- Add support for multiple object classification models
- Add support for multiple object tracking models

Author: Dmitri Lyalikov
"""


import cv2
import numpy as np
from util import *

import time
import subprocess
import signal
import os
import pickle
from colorama import Fore, Style
from math import *
import argparse

# degrees from straight down the camera faces (relative to UAV)
# positive = angled forward, negative = angled backward
CAM_MOUNT_ANGLE = 50

# which RC channel to read from for starting / stopping recording or detection loops
CONTROL_CHANNEL = '7'

# status messages to send to custom GCS
STATUS_RECORDING_START = b'REC_START'
STATUS_RECORDING_STOP = b'REC_STOP'
STATUS_DETECTION_START = b'DET_START'
STATUS_DETECTION_STOP = b'DET_STOP'

class App:
    """
    Represents the main application for UAV MobFinder.

    Attributes:
        args (argparse.Namespace): Command line arguments.
        vehicle (dronekit.Vehicle): The connected vehicle.
        record_process (subprocess.Popen): The process for video recording.
        telemetry_file (file): The file for storing telemetry data.
        running (bool): Flag indicating if the application is running.
    """

    def __init__(self):
        """
        Initializes the App class.
        """
        # parse command line arguments
        

    def start_video_capture(self, file_name):
        """
        Starts the video capture process.

        Args:
            file_name (str): The name of the video file to save the captured video.
        """
        if self.record_process is None:
            # create a gstreamer process to write to the mp4 file 'file_name'
            self.record_process = subprocess.Popen(['gst-launch-1.0', '-e', 'nvarguscamerasrc', 'exposuretimerange="10000000 10000000"', 'maxperf=true', '!' ,'video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1', '!', 'nvvidconv', 'flip-method=2', '!', 'x264enc', 'speed-preset=3', '!', 'mp4mux', '!', 'filesink', f'location={file_name}'], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            while True:
                try:
                    # wait for the video stream to be set to 'PLAYING'
                    line = self.record_process.stdout.readline()
                    if 'PLAYING' in str(line):
                        break
                except:
                    break
            print('Capture started')

    def stop_video_capture(self):
        """
        Stops the video capture process.
        """
        # interrupt the video capture stream, but allow it to finish writing to the mp4 file so it is not corrupted
        if self.record_process is not None:
            self.record_process.send_signal(signal.SIGINT)
            self.record_process.wait()
            self.record_process = None

    def on_sigterm(self, signum, frame):
        """
        Handles the SIGTERM signal.

        Args:
            signum (int): The signal number.
            frame (frame): The current stack frame.
        """
        print('SIGTERM')
        self.running = False
        self.stop_video_capture()

    def run(self):
        """
        Runs the main application.
        If the 'record' flag is set, it enters capture mode and records video and telemetry data.
        If the 'record' flag is not set, it enters detection mode and performs object detection using YOLOv3.
        """
        # code implementation...
        if self.args.record:
            print('[Capture mode]')

            # set to 10W mode so video capture is smooth to mp4
            # will not brown own because capture is less intensive than
            subprocess.call(['nvpmodel', '-m0'])
            
            # setup recording variables
            record_num = 1
            is_recording = False
            recording_t0 = None
            self.running = True

            while self.running:
                
                # check if should record based on value of RC control channel
                control_channel_value = self.vehicle.channels[CONTROL_CHANNEL]
                if control_channel_value is None:
                    time.sleep(1)
                    continue

                should_record = control_channel_value > 1200

                # check if state should change
                if should_record != is_recording:
                    if should_record:
                        # find next available file name
                        while True:
                            vid_file = f'vid{record_num}.mp4'
                            if not os.path.exists(vid_file):
                                break
                            record_num += 1
                        
                        # begin capturing video and telemetry data
                        print(f'Starting capture: {vid_file}')
                        self.start_video_capture(vid_file)
                        self.telemetry_file = open(f'telem{record_num}.txt', 'w')

                    else:
                        # stop video capture and close telemtry file
                        self.stop_video_capture()
                        print('Stopped capture')

                    # update current status and send status message
                    is_recording = should_record
                    self.vehicle.send_mavlink(msg)
            
                if is_recording:
                    # get location, heading, and attitude of the UAV
                    location = self.vehicle.location.global_relative_frame
                    heading = self.vehicle.heading
                    attitude = self.vehicle.attitude
                    
                    if recording_t0 is None:
                        recording_t0 = time.time()

                    # write data to telemetry file with ';' delimeter
                    data_line = f'{time.time()-recording_t0:.3f};{int(location.lat*1e7)};{int(location.lon*1e7)};{location.alt};{heading};{np.degrees(attitude.roll):.3f};{np.degrees(attitude.pitch):.3f}'
                    self.telemetry_file.write(data_line + '\n')
                    time.sleep(0.1)
                else:
                    time.sleep(0.1)
        else:
            
            print('[Detection mode]')
            
            # set Jetson to 5W mode so the Dev kit does not draw too much power and brownout
            subprocess.call(['nvpmodel', '-m1'])

            caldata_path = 'calibration.pkl'

            # load calibration data if it exists
            if os.path.exists(caldata_path):
                cal_data = pickle.load(open(caldata_path, 'rb'))
            else:
                # exit if could not locate the calibration data file
                print(Fore.RED + 'No calibration data! Please run calibrate.py first.' + Style.RESET_ALL)
                return

            # load YOLOv3 network and weights needed for object detection
            net = load_net(b'model/eagleeye.cfg', b'model/eagleeye.weights', 0)
            meta = load_meta(b'model/eagleeye.data')

            # open the video capture pipeline using a gstreamer pipeline string
            pipeline = gstreamer_pipeline(capture_size=(1280, 720), display_size=(1280, 720), framerate=60, flip_method=0)        
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            # exit if the video stream could not be opened
            if not cap.isOpened():
                print(Fore.RED + 'Could not open video stream!' + Style.RESET_ALL)
                return
            
            is_detecting = False
            self.running = True

            while self.running:
                try:

                    control_channel_value = self.vehicle.channels[CONTROL_CHANNEL]
                    if control_channel_value is None:
                        print('No control')
                        time.sleep(1)
                        continue

                    should_detect = control_channel_value > 1200
                    if should_detect != is_detecting:
                        if should_detect:
                            print('Detection started')
                            msg = self.vehicle.message_factory.statustext_encode(6, STATUS_DETECTION_START)
                        else:
                            print('Detection stopped')
                            msg = self.vehicle.message_factory.statustext_encode(6, STATUS_DETECTION_STOP)
                        
                        is_detecting = should_detect
                        self.vehicle.send_mavlink(msg)

                    if not is_detecting:
                        continue

                    # get location, heading, and attitude of the UAV just before the camera frame is captured
                    location = self.vehicle.location.global_relative_frame
                    heading = self.vehicle.heading
                    attitude = self.vehicle.attitude

                    # get the camera frame from the capture stream
                    _, img = cap.read()
                    img_w, img_h = img.shape[1], img.shape[0]

                    # run YOLOv7 detection on the frame using the model loaded at startup
                    objs = detect(net, meta, img, thresh=0.3)

                    # list of center points of all objects detected
                    pts = []

                    # add the center point of each object to the list
                    for obj in objs:
                        classifier, probability, bbox = obj
                        x, y, w, h = bbox
                        pts.append([x, y])

                    # only do the undistortion and transformation calculations if there were any objects detected
                    # to speed up the detection loop
                    if len(pts) > 0:

                       
                        
                        # generate a vector from the focal point to the camera plane for each detected object (x,y,z = right,back,down) relative to the UAV
                        obj_vectors = np.hstack((pts_undist, np.ones((pts_undist.shape[0], 1))))
                        
                        # calculate transformation matrix to orient points in the camera in a North-East-Down reference frame
                        # corrects for roll, pitch, and heading of the UAV
                        mat_transform = matrix_rot_y(attitude.roll) @ matrix_rot_x(-attitude.pitch - CAM_MOUNT_ANGLE * pi / 180) @ matrix_rot_z(-(90 + heading) * pi / 180)

                        for i, obj in enumerate(objs):
                            classifier, probability, bbox = obj
                            obj_vec = obj_vectors[i]

                            # transform the vector toward the object to be in a North-East-Down reference frame relative to the UAV
                            ned_vec = obj_vec @ mat_transform
                            # approximate lattitude and longitude by extending the object's vector from the location and altitude of the UAV down to the ground
                            obj_lat = location.lat + location.alt * (ned_vec[0] / ned_vec[2]) * DEGREES_PER_METER
                            obj_lon = location.lon + location.alt * (ned_vec[1] / ned_vec[2]) * DEGREES_PER_METER

                            msg = self.vehicle.message_factory.command_int_encode(255, 1, 0, 31000, 0, 0, probability, 0, 0, 0, int(obj_lat * 1e7), int(obj_lon * 1e7), 0)
                            self.vehicle.send_mavlink(msg)
                            print(f'[Detect] {probability:.2%}, LAT: {obj_lat:.7f}, LON: {obj_lon:.7f}')

                except KeyboardInterrupt:
                    break
                
    def stop(self):
        # close the connection to the Pixhawk before stopping the script
        print('Closing connection...')
        self.vehicle.close()



def main():
    app = App()
    app.run()
    app.stop()

if __name__ == '__main__':
    main()